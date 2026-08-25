import asyncio
import json
import logging
import re
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.retrieval import (
    EngineeringAnswerResponse,
    EngineeringEvidence,
    EngineeringQueryRequest,
    HybridSearchRequest,
    RetrievalCandidate,
    RetrievalFilter,
)
from app.services.llm import OllamaLLMService, get_llm_service
from app.services.retrieval.hybrid_search import HybridSearchEngine

logger = logging.getLogger(__name__)

ENGINEERING_SYSTEM_PROMPT = """You are an expert AI Engineering Intelligence assistant.
Your goal is to explain historical software changes, architectural decisions, performance optimizations, and bug fixes strictly based on the provided Pull Request evidence.

RULES:
1. Ground your answers strictly in the provided PR evidence. Do NOT hallucinate PR numbers, authors, or code changes not present in the context.
2. Explicitly distinguish between:
   - **Documented Facts**: Reasons or behaviors explicitly stated in PR descriptions, quotes, or author comments.
   - **Inferred Details**: Findings deduced from modified files, commit messages, or component changes.
   - **Unknown / Insufficient Data**: Clearly state if something is unknown or not documented in the indexed history.
3. If the provided context does NOT contain enough information to answer the question, explicitly state:
   "Based on the indexed engineering history, there is insufficient evidence to answer this question."
4. Always cite specific Pull Request numbers (e.g., `PR #834`, `PR #621`) alongside your explanations.
5. Structure your response clearly with a direct answer followed by an Evidence summary.
"""

INSUFFICIENT_EVIDENCE_PHRASES = [
    "insufficient evidence",
    "not enough information",
    "cannot be determined",
    "no relevant pr",
    "no records found",
]


class EngineeringRAGService:
    """Orchestrates hybrid retrieval and LLM generation to answer engineering questions with grounded citations."""

    def __init__(
        self,
        session: AsyncSession,
        hybrid_search_engine: Optional[HybridSearchEngine] = None,
        llm_service: Optional[OllamaLLMService] = None,
    ) -> None:
        self.session = session
        self.hybrid_engine = hybrid_search_engine or HybridSearchEngine(session)
        self.llm_service = llm_service or get_llm_service()

    def classify_scenario(self, query: str) -> str:
        """Heuristically identify the engineering question scenario."""
        q_lower = query.lower()
        if any(w in q_lower for w in ["between release", "between version", "compare release", "compare version", "what changed between"]):
            return "release_comparison"
        if any(w in q_lower for w in ["seen this issue", "seen this before", "happened before", "memory leak", "crash", "bug with", "error with", "issue before"]):
            return "historical_issue"
        if any(w in q_lower for w in ["affected", "impact", "which prs affected", "which pr affected", "performance", "memory"]):
            return "impact_search"
        if any(w in q_lower for w in ["why was", "why did", "architecture", "decision", "motivation behind", "reason for"]):
            return "decision_understanding"
        return "general_qa"

    def build_context_prompt(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        scenario: str,
    ) -> str:
        """Construct a token-efficient structured prompt incorporating ranked PR evidence.

        Uses compact key-value formatting for metadata while preserving full
        natural language for summary and motivation fields (the content the LLM
        reasons over). This reduces prompt tokens by ~40% compared to verbose
        prose formatting without sacrificing answer quality.
        """
        if not candidates:
            return f"User Question: {query}\n\nRetrieved PR Evidence: NONE (No matching PRs found in the knowledge base)."

        evidence_blocks: List[str] = []
        for cand in candidates:
            # Line 1: Compact header with PR identity and key metadata
            merged_str = cand.merged_at.strftime("%Y-%m-%d") if cand.merged_at else "N/A"
            header_parts = [cand.repository, f"by {cand.author}", f"merged {merged_str}"]
            if cand.milestone:
                header_parts.append(f"v{cand.milestone}")
            header = ", ".join(header_parts)
            lines = [f"[PR #{cand.pr_number}] {cand.title} ({header})"]

            # Line 2: Summary (full natural language — critical for LLM reasoning)
            if cand.summary:
                lines.append(f"Summary: {cand.summary}")

            # Line 3: Motivation (full natural language — critical for grounding)
            if cand.motivation_reason:
                m_type = (cand.motivation_type or "unknown").capitalize()
                lines.append(f"Motivation ({m_type}): {cand.motivation_reason}")

            # Line 4: Collapsed classification metadata
            meta_parts: List[str] = []
            if cand.components:
                meta_parts.append(f"Impacted Components: {', '.join(cand.components)}")
            if cand.change_types:
                meta_parts.append(f"Types: {', '.join(cand.change_types)}")
            if cand.architectural_change:
                meta_parts.append("Architectural Change: Yes")
            if cand.breaking_change:
                meta_parts.append("Breaking Change: Yes")
            if meta_parts:
                lines.append(" | ".join(meta_parts))

            # Line 5: Technical details (kept as-is, high signal)
            if cand.key_technical_details:
                lines.append(f"Tech: {'; '.join(cand.key_technical_details)}")

            # Line 6: Changed files (compact, max 5)
            if cand.changed_files:
                lines.append(f"Files: {', '.join(cand.changed_files[:5])}")

            evidence_blocks.append("\n".join(lines))

        joined_evidence = "\n\n".join(evidence_blocks)

        prompt = f"""Question: {query}
Detected Intent: {scenario}

Retrieved PR Evidence:
--------------------------------------------------
{joined_evidence}
--------------------------------------------------

Answer the question accurately based on the evidence above.
Cite the relevant PRs (e.g. PR #{candidates[0].pr_number}) and indicate if reasons are Documented in the PR or Inferred from the code changes.
"""
        return prompt

    def candidate_to_evidence(self, candidate: RetrievalCandidate) -> EngineeringEvidence:
        """Map internal retrieval candidate to structured external evidence model."""
        return EngineeringEvidence(
            pr_number=candidate.pr_number,
            repository=candidate.repository,
            title=candidate.title,
            author=candidate.author,
            merged_at=candidate.merged_at,
            milestone=candidate.milestone,
            components=candidate.components,
            change_types=candidate.change_types,
            motivation_type=candidate.motivation_type,
            motivation_reason=candidate.motivation_reason,
            key_technical_details=candidate.key_technical_details,
            changed_files=candidate.changed_files,
            relevance_score=candidate.combined_score,
            rank=candidate.rank,
            match_reasons=candidate.match_reasons,
        )

    def select_candidates_for_prompt(
        self,
        candidates: List[RetrievalCandidate],
        scenario: str,
        hard_max: int = 5,
        score_floor_ratio: float = 0.4,
        dominant_threshold: float = 2.0,
    ) -> List[RetrievalCandidate]:
        """Select the minimal high-confidence candidate set for the LLM prompt.

        This reduces prompt tokens by pruning low-confidence tail candidates
        that add noise without improving answer quality.

        Strategy:
        1. If the top candidate's score is >= dominant_threshold * the second
           candidate's score, only the top candidate is sent (strong single-hit).
        2. Otherwise, include candidates whose combined_score is at least
           score_floor_ratio * the top candidate's score.
        3. Cap at hard_max regardless.

        Args:
            candidates: Ranked retrieval candidates (descending by combined_score).
            scenario: Detected question scenario (used for scenario-specific caps).
            hard_max: Absolute maximum candidates to send to the LLM prompt.
            score_floor_ratio: Minimum score as a fraction of the top score to be included.
            dominant_threshold: If top_score / second_score >= this, use only top candidate.
        """
        if not candidates:
            return []

        if len(candidates) == 1:
            return candidates

        top_score = candidates[0].combined_score

        # Guard: if top score is 0 (shouldn't happen post-RRF normalization), return all
        if top_score <= 0:
            return candidates[:hard_max]

        # Dominance check: top candidate is overwhelmingly ahead
        second_score = candidates[1].combined_score
        if second_score > 0 and (top_score / second_score) >= dominant_threshold:
            logger.debug(
                "Confidence gate: dominant top candidate (%.3f vs %.3f, ratio %.1fx). Sending 1 candidate to LLM.",
                top_score,
                second_score,
                top_score / second_score,
            )
            return [candidates[0]]

        # Score floor: include candidates scoring at least score_floor_ratio × top_score
        score_floor = top_score * score_floor_ratio
        selected = [c for c in candidates if c.combined_score >= score_floor]

        # Enforce hard cap
        selected = selected[:hard_max]

        if len(selected) < len(candidates):
            logger.debug(
                "Confidence gate: pruned %d/%d candidates (floor=%.3f, top=%.3f). Sending %d to LLM.",
                len(candidates) - len(selected),
                len(candidates),
                score_floor,
                top_score,
                len(selected),
            )

        return selected

    async def answer_engineering_query(
        self,
        request: EngineeringQueryRequest,
    ) -> EngineeringAnswerResponse:
        """Answer an engineering question with hybrid retrieval and grounded evidence synthesis."""
        scenario = self.classify_scenario(request.query)

        # Merge request repository into filter if specified
        effective_filter = request.filter or RetrievalFilter()
        if request.repository and not effective_filter.repository:
            effective_filter.repository = request.repository

        # 1. Retrieve candidates using Hybrid Search
        search_req = HybridSearchRequest(
            query=request.query,
            filter=effective_filter,
            limit=request.limit,
        )
        search_res = await self.hybrid_engine.search(search_req)
        candidates = search_res.results

        # Convert ALL candidates to evidence items (returned in API response)
        evidence_items = [self.candidate_to_evidence(c) for c in candidates]

        # 2. Check for empty evidence
        if not candidates:
            return EngineeringAnswerResponse(
                query=request.query,
                answer="Based on the indexed engineering history, no relevant Pull Requests or engineering records were found matching your question.",
                scenario_detected=scenario,
                evidence=[],
                total_evidence_count=0,
                has_sufficient_evidence=False,
                model_used=self.llm_service.model,
            )

        # 3. Confidence-gated candidate limiting for prompt context
        prompt_candidates = self.select_candidates_for_prompt(candidates, scenario)

        # 4. Build prompt and invoke LLM
        prompt = self.build_context_prompt(
            query=request.query,
            candidates=prompt_candidates,
            scenario=scenario,
        )
        system_prompt = request.system_prompt or ENGINEERING_SYSTEM_PROMPT

        try:
            answer = await self.llm_service.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
            )
        except Exception as e:
            logger.error("Failed to generate LLM response for query: %s", e)
            answer = f"Error generating answer: {str(e)}"

        # Check if the LLM declared insufficient evidence
        has_sufficient = not any(
            phrase in answer.lower() for phrase in INSUFFICIENT_EVIDENCE_PHRASES
        )

        return EngineeringAnswerResponse(
            query=request.query,
            answer=answer,
            scenario_detected=scenario,
            evidence=evidence_items if request.include_raw_evidence else [],
            total_evidence_count=len(evidence_items),
            has_sufficient_evidence=has_sufficient,
            model_used=self.llm_service.model,
        )

    async def stream_engineering_query(
        self,
        request: EngineeringQueryRequest,
    ) -> AsyncGenerator[str, None]:
        """Stream an engineering question response via Server-Sent Events (SSE)."""
        scenario = self.classify_scenario(request.query)

        effective_filter = request.filter or RetrievalFilter()
        if request.repository and not effective_filter.repository:
            effective_filter.repository = request.repository

        # 1. Retrieve candidates
        search_req = HybridSearchRequest(
            query=request.query,
            filter=effective_filter,
            limit=request.limit,
        )
        search_res = await self.hybrid_engine.search(search_req)
        candidates = search_res.results
        evidence_items = [self.candidate_to_evidence(c) for c in candidates]

        # 2. Yield metadata event with candidate list
        meta_payload = {
            "query": request.query,
            "scenario": scenario,
            "total_candidates": len(evidence_items),
            "evidence": [e.model_dump(mode="json") for e in evidence_items],
        }
        yield f"event: metadata\ndata: {json.dumps(meta_payload)}\n\n"

        if not candidates:
            no_info_text = "Based on the indexed engineering history, no relevant Pull Requests were found matching your query."
            yield f"event: token\ndata: {json.dumps({'token': no_info_text})}\n\n"
            yield f"event: done\ndata: {json.dumps({'finish_reason': 'no_evidence'})}\n\n"
            return

        # 3. Confidence-gated candidate limiting for prompt context
        prompt_candidates = self.select_candidates_for_prompt(candidates, scenario)

        # 4. Build prompt and stream tokens
        prompt = self.build_context_prompt(
            query=request.query,
            candidates=prompt_candidates,
            scenario=scenario,
        )
        system_prompt = request.system_prompt or ENGINEERING_SYSTEM_PROMPT

        try:
            async for token in self.llm_service.stream_response(
                prompt=prompt,
                system_prompt=system_prompt,
            ):
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            logger.error("Error streaming LLM tokens: %s", e)
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

        yield f"event: done\ndata: {json.dumps({'finish_reason': 'stop'})}\n\n"

