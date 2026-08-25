import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.domain.models import ChangedFile, Commit, PRUnderstanding, PullRequest, Repository
from app.domain.retrieval import (
    HybridSearchRequest,
    HybridSearchResponse,
    RetrievalCandidate,
    RetrievalFilter,
)
from app.services.embedding import EmbeddingService, get_embedding_service
from app.services.vector_store import VectorStoreManager, get_vector_store

logger = logging.getLogger(__name__)

# Standard stop words to ignore during lexical token extraction
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
    "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've",
    "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more",
    "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves", "tell", "show", "find", "give", "list", "which", "prs", "pr"
}


class HybridSearchEngine:
    """Combines dense vector semantic search, exact keyword/lexical search, metadata filtering,
    and Reciprocal Rank Fusion (RRF) to retrieve relevant engineering history."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStoreManager] = None,
    ) -> None:
        self.session = session
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_vector_store()

    def extract_search_tokens(self, query: str) -> List[str]:
        """Extract meaningful keywords, file names, code identifiers, and PR numbers from the query."""
        raw_tokens = re.findall(r"[A-Za-z0-9_.\-#]+", query)
        filtered_tokens: List[str] = []
        for t in raw_tokens:
            cleaned = t.strip("#").lower()
            if not cleaned:
                continue
            # Keep if not a stop word or if it contains code patterns (e.g. .cpp, .py, camelCase)
            if cleaned not in STOP_WORDS or "." in t or "_" in t:
                filtered_tokens.append(t)
        return filtered_tokens

    async def search_vectors(
        self,
        query: str,
        retrieval_filter: Optional[RetrievalFilter] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Perform semantic similarity search over dense vector embeddings in ChromaDB."""
        # 1. Embed query (run non-blocking)
        query_embedding = await asyncio.to_thread(self.embedding_service.embed_text, query)

        # 2. Build ChromaDB metadata filter
        where_conditions: Dict[str, Any] = {}
        if retrieval_filter:
            if retrieval_filter.repository:
                where_conditions["repository"] = retrieval_filter.repository
            elif retrieval_filter.owner and retrieval_filter.repo:
                where_conditions["repository"] = f"{retrieval_filter.owner}/{retrieval_filter.repo}"

            if retrieval_filter.architectural_only:
                where_conditions["architectural_change"] = True

            if retrieval_filter.breaking_only:
                where_conditions["breaking_change"] = True

        where_clause = where_conditions if where_conditions else None

        # 3. Query ChromaDB
        try:
            results = await asyncio.to_thread(
                self.vector_store.query,
                query_embedding=query_embedding,
                n_results=limit * 2,
                where=where_clause,
            )
        except Exception as e:
            logger.warning("Vector search encountered an error: %s", e)
            return []

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        vector_hits: List[Dict[str, Any]] = []
        for doc_id, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
            if not meta:
                continue
            similarity = max(0.0, min(1.0, 1.0 - (dist if dist is not None else 1.0)))
            repo_name = meta.get("repository", "")
            pr_num = meta.get("pr_number")
            if not repo_name or not pr_num:
                continue

            vector_hits.append(
                {
                    "doc_id": doc_id,
                    "repository": repo_name,
                    "pr_number": int(pr_num),
                    "vector_score": similarity,
                    "doc_text": doc_text,
                    "metadata": meta,
                }
            )

        # Sort descending by vector similarity score
        vector_hits.sort(key=lambda x: x["vector_score"], reverse=True)
        return vector_hits[:limit]

    async def search_keywords(
        self,
        query: str,
        retrieval_filter: Optional[RetrievalFilter] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Perform exact keyword, file name, and lexical token search across PR records in the database."""
        tokens = self.extract_search_tokens(query)
        if not tokens:
            return []

        stmt = (
            select(PullRequest, Repository, PRUnderstanding)
            .join(Repository, PullRequest.repository_id == Repository.id)
            .outerjoin(PRUnderstanding, PRUnderstanding.pull_request_id == PullRequest.id)
            .options(
                selectinload(PullRequest.changed_files),
                selectinload(PullRequest.commits),
            )
        )

        # Apply database filters
        if retrieval_filter:
            if retrieval_filter.repository:
                if "/" in retrieval_filter.repository:
                    owner, name = retrieval_filter.repository.split("/", 1)
                    stmt = stmt.where(Repository.owner == owner, Repository.name == name)
            elif retrieval_filter.owner and retrieval_filter.repo:
                stmt = stmt.where(
                    Repository.owner == retrieval_filter.owner,
                    Repository.name == retrieval_filter.repo,
                )

            if retrieval_filter.author:
                stmt = stmt.where(PullRequest.author.ilike(f"%{retrieval_filter.author}%"))

            if retrieval_filter.milestone:
                stmt = stmt.where(PullRequest.milestone == retrieval_filter.milestone)

            if retrieval_filter.since:
                stmt = stmt.where(PullRequest.merged_at >= retrieval_filter.since)

            if retrieval_filter.until:
                stmt = stmt.where(PullRequest.merged_at <= retrieval_filter.until)

            if retrieval_filter.architectural_only:
                stmt = stmt.where(PRUnderstanding.architectural_change == True)  # noqa: E712

            if retrieval_filter.breaking_only:
                stmt = stmt.where(PRUnderstanding.breaking_change == True)  # noqa: E712

        # Fetch candidate PRs
        res = await self.session.execute(stmt)
        rows = res.all()

        keyword_hits: List[Dict[str, Any]] = []

        query_lower = query.lower()
        pr_number_match = re.search(r"#?(\d+)", query)
        target_pr_number = int(pr_number_match.group(1)) if pr_number_match else None

        for pr, repo, understanding in rows:
            score = 0.0
            matched_aspects: List[str] = []

            # 1. Exact PR number match boost
            if target_pr_number and pr.number == target_pr_number:
                score += 10.0
                matched_aspects.append(f"PR #{pr.number} exact number match")

            pr_title = pr.title or ""
            pr_title_lower = pr_title.lower()
            pr_body = pr.body or ""
            pr_body_lower = pr_body.lower()

            # 2. Exact multi-word phrase matching
            if len(query_lower) > 3 and query_lower in pr_title_lower:
                score += 5.0
                matched_aspects.append(f"Exact phrase match in title: '{query}'")
            elif understanding and understanding.summary and query_lower in understanding.summary.lower():
                score += 4.0
                matched_aspects.append(f"Exact phrase match in summary: '{query}'")

            # 3. Token-by-token scoring
            token_matches = 0
            for token in tokens:
                t_lower = token.lower()
                token_matched = False

                # PR Title (high weight)
                if t_lower in pr_title_lower:
                    score += 3.0
                    token_matched = True

                # AI Understanding Summary & Motivation (high weight)
                if understanding:
                    if understanding.summary and t_lower in understanding.summary.lower():
                        score += 2.5
                        token_matched = True
                    if understanding.motivation_reason and t_lower in understanding.motivation_reason.lower():
                        score += 2.0
                        token_matched = True
                    if understanding.components:
                        for comp in understanding.components:
                            if t_lower in comp.lower():
                                score += 2.5
                                token_matched = True
                    if understanding.change_types:
                        for ct in understanding.change_types:
                            if t_lower == ct.lower():
                                score += 2.0
                                token_matched = True
                    if understanding.key_technical_details:
                        for tech in understanding.key_technical_details:
                            if t_lower in tech.lower():
                                score += 2.0
                                token_matched = True

                # Changed File Names (high weight for filenames/extensions like ImageCache.cpp)
                if pr.changed_files:
                    for cf in pr.changed_files:
                        if t_lower in cf.filename.lower():
                            score += 3.0
                            token_matched = True
                            matched_aspects.append(f"File match: {cf.filename}")
                            break

                # PR Body / description
                if t_lower in pr_body_lower:
                    score += 1.0
                    token_matched = True

                if token_matched:
                    token_matches += 1

            if score > 0:
                coverage_ratio = token_matches / len(tokens)
                final_lexical_score = score * (1.0 + coverage_ratio)
                repo_full_name = f"{repo.owner}/{repo.name}"

                keyword_hits.append(
                    {
                        "repository": repo_full_name,
                        "pr_number": pr.number,
                        "keyword_score": final_lexical_score,
                        "matched_aspects": matched_aspects,
                        "pr_record": pr,
                        "repo_record": repo,
                        "understanding_record": understanding,
                    }
                )

        # Sort descending by keyword score
        keyword_hits.sort(key=lambda x: x["keyword_score"], reverse=True)
        return keyword_hits[:limit]

    async def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        """Execute hybrid search combining dense vector similarity, lexical matching, and RRF rank fusion."""
        fetch_limit = max(request.limit * 2, 20)

        # 1. Concurrently / sequentially run Vector Search and Keyword Search
        vector_results = await self.search_vectors(
            query=request.query,
            retrieval_filter=request.filter,
            limit=fetch_limit,
        )

        keyword_results = await self.search_keywords(
            query=request.query,
            retrieval_filter=request.filter,
            limit=fetch_limit,
        )

        # 2. Reciprocal Rank Fusion (RRF)
        # Map: (repo, pr_number) -> candidate dict
        candidates_map: Dict[Tuple[str, int], Dict[str, Any]] = {}

        # Process Vector Rankings
        for rank_v, hit in enumerate(vector_results, start=1):
            key = (hit["repository"], hit["pr_number"])
            if key not in candidates_map:
                candidates_map[key] = {
                    "repository": hit["repository"],
                    "pr_number": hit["pr_number"],
                    "vector_score": hit["vector_score"],
                    "keyword_score": None,
                    "vector_rank": rank_v,
                    "keyword_rank": None,
                    "match_reasons": [f"Semantic vector similarity match ({hit['vector_score']:.2f})"],
                    "metadata": hit.get("metadata", {}),
                }
            else:
                candidates_map[key]["vector_score"] = hit["vector_score"]
                candidates_map[key]["vector_rank"] = rank_v
                candidates_map[key]["match_reasons"].append(
                    f"Semantic vector similarity match ({hit['vector_score']:.2f})"
                )

        # Process Keyword Rankings
        for rank_k, hit in enumerate(keyword_results, start=1):
            key = (hit["repository"], hit["pr_number"])
            aspects_summary = ", ".join(hit.get("matched_aspects", [])[:2])
            reason = f"Keyword match ({hit['keyword_score']:.1f} pts)"
            if aspects_summary:
                reason += f": {aspects_summary}"

            if key not in candidates_map:
                candidates_map[key] = {
                    "repository": hit["repository"],
                    "pr_number": hit["pr_number"],
                    "vector_score": None,
                    "keyword_score": hit["keyword_score"],
                    "vector_rank": None,
                    "keyword_rank": rank_k,
                    "match_reasons": [reason],
                    "pr_record": hit.get("pr_record"),
                    "repo_record": hit.get("repo_record"),
                    "understanding_record": hit.get("understanding_record"),
                }
            else:
                candidates_map[key]["keyword_score"] = hit["keyword_score"]
                candidates_map[key]["keyword_rank"] = rank_k
                candidates_map[key]["match_reasons"].append(reason)
                if "pr_record" not in candidates_map[key]:
                    candidates_map[key]["pr_record"] = hit.get("pr_record")
                    candidates_map[key]["repo_record"] = hit.get("repo_record")
                    candidates_map[key]["understanding_record"] = hit.get("understanding_record")

        if not candidates_map:
            return HybridSearchResponse(
                query=request.query,
                total_candidates=0,
                results=[],
                vector_hits=len(vector_results),
                keyword_hits=len(keyword_results),
            )

        # 3. Calculate RRF Score for each candidate
        k = request.rrf_k
        w_v = request.vector_weight
        w_k = request.keyword_weight

        scored_candidates: List[Dict[str, Any]] = []
        for (repo_name, pr_num), item in candidates_map.items():
            r_v = item["vector_rank"]
            r_k = item["keyword_rank"]

            rrf_score = 0.0
            if r_v is not None:
                rrf_score += w_v * (1.0 / (k + r_v))
            if r_k is not None:
                rrf_score += w_k * (1.0 / (k + r_k))

            # Bonus when item is present in both vector and keyword search
            if r_v is not None and r_k is not None:
                rrf_score *= 1.15
                item["match_reasons"].insert(0, "Dual-modality match (Found in both Semantic & Keyword search)")

            item["combined_score"] = rrf_score
            scored_candidates.append(item)

        # Sort descending by fused RRF score
        scored_candidates.sort(key=lambda x: x["combined_score"], reverse=True)
        top_candidates = scored_candidates[: request.limit]

        # 4. Normalize RRF scores to 0.0 - 1.0 range for readable output
        max_score = top_candidates[0]["combined_score"] if top_candidates else 1.0
        if max_score > 0:
            for c in top_candidates:
                c["combined_score"] = round(c["combined_score"] / max_score, 4)

        # 5. Hydrate candidates with full PullRequest & PRUnderstanding domain details
        hydrated_results: List[RetrievalCandidate] = []
        for rank_idx, cand in enumerate(top_candidates, start=1):
            pr_rec = cand.get("pr_record")
            repo_rec = cand.get("repo_record")
            und_rec = cand.get("understanding_record")

            if not pr_rec:
                # Load PR from DB if only retrieved via vector store
                repo_owner, repo_name = cand["repository"].split("/", 1)
                stmt = (
                    select(PullRequest, Repository, PRUnderstanding)
                    .join(Repository, PullRequest.repository_id == Repository.id)
                    .outerjoin(PRUnderstanding, PRUnderstanding.pull_request_id == PullRequest.id)
                    .options(
                        selectinload(PullRequest.changed_files),
                        selectinload(PullRequest.commits),
                    )
                    .where(
                        Repository.owner == repo_owner,
                        Repository.name == repo_name,
                        PullRequest.number == cand["pr_number"],
                    )
                )
                res = await self.session.execute(stmt)
                row = res.first()
                if row:
                    pr_rec, repo_rec, und_rec = row

            if not pr_rec:
                # Fallback to metadata if DB lookup failed
                meta = cand.get("metadata", {})
                candidate_obj = RetrievalCandidate(
                    pr_number=cand["pr_number"],
                    repository=cand["repository"],
                    title=meta.get("pr_title", f"PR #{cand['pr_number']}"),
                    author=meta.get("author", "unknown"),
                    state=meta.get("state", "closed"),
                    merged_at=datetime.fromisoformat(meta["merged_at"]) if meta.get("merged_at") else None,
                    milestone=meta.get("milestone"),
                    summary=None,
                    vector_score=cand.get("vector_score"),
                    keyword_score=cand.get("keyword_score"),
                    combined_score=cand["combined_score"],
                    rank=rank_idx,
                    match_reasons=cand.get("match_reasons", []),
                )
            else:
                files_list = [f.filename for f in pr_rec.changed_files] if pr_rec.changed_files else []
                candidate_obj = RetrievalCandidate(
                    pr_number=pr_rec.number,
                    repository=cand["repository"],
                    title=pr_rec.title,
                    author=pr_rec.author,
                    state=pr_rec.state,
                    merged_at=pr_rec.merged_at,
                    milestone=pr_rec.milestone,
                    summary=und_rec.summary if und_rec else None,
                    motivation_reason=und_rec.motivation_reason if und_rec else None,
                    motivation_type=und_rec.motivation_type if und_rec else None,
                    components=und_rec.components if (und_rec and und_rec.components) else [],
                    change_types=und_rec.change_types if (und_rec and und_rec.change_types) else [],
                    architectural_change=bool(und_rec.architectural_change) if und_rec else False,
                    breaking_change=bool(und_rec.breaking_change) if und_rec else False,
                    key_technical_details=und_rec.key_technical_details if (und_rec and und_rec.key_technical_details) else [],
                    changed_files=files_list[:10],
                    vector_score=cand.get("vector_score"),
                    keyword_score=cand.get("keyword_score"),
                    combined_score=cand["combined_score"],
                    rank=rank_idx,
                    match_reasons=cand.get("match_reasons", []),
                )

            hydrated_results.append(candidate_obj)

        return HybridSearchResponse(
            query=request.query,
            total_candidates=len(hydrated_results),
            results=hydrated_results,
            vector_hits=len(vector_results),
            keyword_hits=len(keyword_results),
        )
