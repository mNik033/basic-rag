import json
import logging
import re
from typing import Any, Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.domain.models import (
    ChangedFile,
    Commit,
    PRUnderstanding,
    PullRequest,
    Repository,
    Review,
    ReviewComment,
)
from app.domain.understanding import (
    EvidenceType,
    MotivationDetail,
    PRUnderstandingProcessRequest,
    PRUnderstandingProcessResponse,
    PRUnderstandingResult,
)
from app.services.llm import OllamaLLMService, get_llm_service

logger = logging.getLogger(__name__)

PR_UNDERSTANDING_SYSTEM_PROMPT = """You are an expert Staff Software Architect and Engineering Intelligence Analyst.
Analyze the provided Pull Request (description, commits, diffs, reviews) and generate a structured engineering summary.

RULES:
1. "summary": Exactly 1-2 concise sentences summarizing what changed. Do NOT use bullet points, lists, or newlines in the summary.
2. "motivation": Classify into "documented" (explicitly stated in text/comments with a short exact "evidence_quote"), "inferred" (deduced from code), or "unknown".
3. "components": List at most 3-4 affected module/subsystem/package names.
4. "change_types": Select relevant tags from: ["feature", "bugfix", "performance", "memory", "refactor", "security", "api-change", "docs", "translation", "dependencies", "ci-cd"].
5. "impact": 1-3 concise phrases describing the technical impact.
6. "architectural_change": true only if introducing/modifying core system architecture or design patterns.
7. "breaking_change": true if backward-incompatible.
8. "key_technical_details": 1-3 brief technical highlights (e.g. algorithms, data structures, modified APIs).

Respond ONLY with the requested JSON structure. Never output conversational text or markdown formatting around the JSON.
"""


# Patterns for noise / non-source files that should not consume LLM diff context
IGNORED_DIFF_PATTERNS = [
    # Lockfiles
    r"(^|/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|Pipfile\.lock|Cargo\.lock|go\.sum|composer\.lock)$",
    # Minified assets & source maps
    r"\.(min\.js|min\.css|map|wasm)$",
    # Generated docs / dist directories
    r"(^|/)(dist|build|\.next|out|coverage|\.pytest_cache)/",
    # Binary / image / font files
    r"\.(png|jpe?g|gif|ico|svg|webp|avif|ttf|woff2?|eot|pdf|zip|tar|gz|bin)$",
]

# File extensions given priority when budgeting diff context
HIGH_PRIORITY_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java",
    ".cpp", ".c", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".kt", ".scala", ".proto", ".graphql", ".sql", ".yaml", ".yml", ".toml"
}


# Patterns for bot / service account authors
BOT_AUTHOR_PATTERNS = [
    r"\[bot\]$",
    r"\[app\]$",
    r"[-_]bot$",
    r"^(codecov|coveralls|dependabot|renovate|github-actions|pre-commit-ci|vercel|netlify|sonarcloud|imgbot|fastapi-people)$",
]

# Patterns for automated / boilerplate review and comment content
BOT_CONTENT_PATTERNS = [
    r"(?i)coverage\s+(report|decreased|increased|diff|has\s+not\s+been\s+reported)",
    r"(?i)coveralls\.io",
    r"(?i)contributor\s+license\s+agreement",
    r"(?i)cla\s+(signed|check|assistant)",
    r"(?i)preview\s+(url|deployment|ready|available)",
    r"(?i)bundle\s+size\s+report",
    r"(?i)benchmark\s+results",
    r"(?i)this\s+pull\s+request\s+has\s+been\s+automatically\s+marked\s+as\s+stale",
]

# Trivial single-phrase responses that add zero architectural insight
TRIVIAL_COMMENT_PATTERNS = [
    r"^(?i)(lgtm!?|\+1|👍|looks good to me!?|done|fixed|applied|nit:?|thanks!?|thx!?|thank you!?)$"
]


def is_noise_file(filename: str) -> bool:
    """Check if file is a lockfile, minified asset, binary, or build output."""
    norm = filename.replace("\\", "/")
    return any(re.search(pat, norm, re.IGNORECASE) for pat in IGNORED_DIFF_PATTERNS)


def is_meaningful_review_comment(author: str, body: Optional[str]) -> bool:
    """Filter out bot comments, CI reports, CLA notices, and trivial responses."""
    if not body or not body.strip():
        return False

    clean_body = body.strip()

    # 1. Author bot check
    author_lower = (author or "").lower()
    if any(re.search(pat, author_lower) for pat in BOT_AUTHOR_PATTERNS):
        return False

    # 2. Content boilerplate / CI report check
    if any(re.search(pat, clean_body) for pat in BOT_CONTENT_PATTERNS):
        return False

    # 3. Trivial response check
    if any(re.match(pat, clean_body) for pat in TRIVIAL_COMMENT_PATTERNS):
        return False

    return True


def file_priority_key(file: ChangedFile) -> tuple[int, int]:
    """Sort files: high-priority code first (0), standard files next (1), sorted by change count."""
    fn = file.filename.lower()
    has_prio_ext = any(fn.endswith(ext) for ext in HIGH_PRIORITY_EXTENSIONS)
    prio_rank = 0 if has_prio_ext else 1
    # Secondary: files with more modifications first
    return (prio_rank, -(file.changes or (file.additions + file.deletions)))


def clean_json_response(raw_text: str) -> str:
    """Strip markdown code blocks or surrounding text to isolate JSON payload."""
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # If surrounded by text, find outer brackets
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)
    return text


class PRUnderstandingService:
    """Service to process raw GitHub PRs with an LLM and generate structured engineering knowledge."""

    def __init__(
        self,
        session: AsyncSession,
        llm_service: Optional[OllamaLLMService] = None,
        max_diff_chars: int = 8000,
    ) -> None:
        self.session = session
        self.llm_service = llm_service or OllamaLLMService(num_predict=1536, num_ctx=4096)
        self.max_diff_chars = max_diff_chars

    def build_pr_context_prompt(self, pr: PullRequest) -> str:
        """Construct prompt containing PR metadata, commits, changed files/diffs, and review discussions."""
        parts: List[str] = [
            f"# PULL REQUEST #{pr.number}: {pr.title}",
            f"Author: {pr.author}",
            f"State: {pr.state} (Merged at: {pr.merged_at or 'N/A'})",
            f"Labels: {', '.join(pr.labels) if pr.labels else 'None'}",
            f"Milestone: {pr.milestone or 'None'}",
            "",
            "## PR Description:",
            pr.body.strip() if pr.body and pr.body.strip() else "(No description provided)",
            "",
        ]

        # Commits
        if pr.commits:
            parts.append("## Commit Messages:")
            for c in pr.commits:
                parts.append(f"- [{c.sha[:7]}] {c.message.strip()} (+{c.additions}/-{c.deletions})")
            parts.append("")

        # Changed Files & Diffs (smart filtered to prioritize high-signal code changes)
        if pr.changed_files:
            parts.append("## Changed Files & Patches:")

            # Separate noise files (lockfiles, minified assets, binaries) from meaningful source files
            source_files: List[ChangedFile] = []
            skipped_noise: List[str] = []

            for f in pr.changed_files:
                if is_noise_file(f.filename):
                    skipped_noise.append(f.filename)
                else:
                    source_files.append(f)

            # Sort source files: core code extensions first, largest modifications first
            source_files.sort(key=file_priority_key)

            current_diff_len = 0
            for f in source_files:
                diff_header = f"### File: {f.filename} ({f.status}, +{f.additions}/-{f.deletions})"
                patch = f.patch_text or "(Binary or diff unavailable)"

                # Truncate large individual patches to preserve budget
                if len(patch) > 1800:
                    patch = patch[:1800] + "\n... [diff truncated for length] ..."

                diff_block = f"{diff_header}\n```diff\n{patch}\n```\n"

                if current_diff_len + len(diff_block) > self.max_diff_chars:
                    parts.append(f"{diff_header} [Full diff omitted to fit context budget]")
                else:
                    parts.append(diff_block)
                    current_diff_len += len(diff_block)

            if skipped_noise:
                parts.append(f"*Note: {len(skipped_noise)} non-source / generated file(s) omitted from diff (e.g. {', '.join(skipped_noise[:3])}).*")

            parts.append("")

        # Reviews & Discussion (filtered for meaningful human feedback)
        meaningful_discussions: List[str] = []
        if pr.reviews:
            for r in pr.reviews:
                if is_meaningful_review_comment(r.author, r.body):
                    meaningful_discussions.append(f"Review by @{r.author} ({r.state}): {r.body.strip()}")
        if pr.review_comments:
            for rc in pr.review_comments:
                if is_meaningful_review_comment(rc.author, rc.body):
                    loc = f" on {rc.path}:{rc.line}" if rc.path else ""
                    meaningful_discussions.append(f"Comment by @{rc.author}{loc}: {rc.body.strip()}")

        if meaningful_discussions:
            parts.append("## Review Comments & Discussion:")
            parts.extend(meaningful_discussions)
            parts.append("")

        parts.append("Please analyze the above Pull Request and provide the structured JSON engineering summary.")
        return "\n".join(parts)

    async def analyze_pull_request(self, pr: PullRequest) -> PRUnderstandingResult:
        """Call LLM with PR context and parse into validated PRUnderstandingResult."""
        prompt = self.build_pr_context_prompt(pr)
        logger.debug("Prompt sent to LLM for PR #%s:\n%s", pr.number, prompt)

        schema = PRUnderstandingResult.model_json_schema()
        raw_response = await self.llm_service.generate_response(
            prompt=prompt,
            system_prompt=PR_UNDERSTANDING_SYSTEM_PROMPT,
            format_json=schema,
        )
        logger.debug("Raw LLM output for PR #%s:\n%s", pr.number, raw_response)

        cleaned = clean_json_response(raw_response)

        try:
            parsed_dict = json.loads(cleaned)
            result = PRUnderstandingResult.model_validate(parsed_dict)
            result.raw_response = parsed_dict
            return result
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning(
                "Failed strict validation on LLM output for PR #%s: %s.\n"
                "=== RAW LLM RESPONSE ===\n%s\n========================",
                pr.number,
                exc,
                raw_response,
            )
            # Fallback healing
            return PRUnderstandingResult(
                summary=f"PR #{pr.number}: {pr.title}",
                motivation=MotivationDetail(
                    evidence_type=EvidenceType.UNKNOWN,
                    reason=pr.title,
                ),
                components=[],
                change_types=["unknown"],
                impact=[],
                architectural_change=False,
                breaking_change=False,
                key_technical_details=[],
                raw_response={"raw_text": raw_response, "parse_error": str(exc)},
            )

        schema = PRUnderstandingResult.model_json_schema()
        raw_response = await self.llm_service.generate_response(
            prompt=prompt,
            system_prompt=PR_UNDERSTANDING_SYSTEM_PROMPT,
            format_json=schema,
        )
        logger.debug("Raw LLM output for PR #%s:\n%s", pr.number, raw_response)

        cleaned = clean_json_response(raw_response)

        try:
            parsed_dict = json.loads(cleaned)
            result = PRUnderstandingResult.model_validate(parsed_dict)
            result.raw_response = parsed_dict
            return result
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning(
                "Failed strict validation on LLM output for PR #%s: %s.\n"
                "=== RAW LLM RESPONSE ===\n%s\n========================",
                pr.number,
                exc,
                raw_response,
            )
            # Fallback healing
            return PRUnderstandingResult(
                summary=f"PR #{pr.number}: {pr.title}",
                motivation=MotivationDetail(
                    evidence_type=EvidenceType.UNKNOWN,
                    reason=pr.title,
                ),
                components=[],
                change_types=["unknown"],
                impact=[],
                raw_response={"raw_text": raw_response, "parse_error": str(exc)},
            )

    async def save_understanding(
        self,
        pr_id: int,
        understanding: PRUnderstandingResult,
        model_name: str,
    ) -> PRUnderstanding:
        """Upsert PRUnderstanding record in PostgreSQL."""
        stmt = select(PRUnderstanding).where(PRUnderstanding.pull_request_id == pr_id)
        res = await self.session.execute(stmt)
        record = res.scalar_one_or_none()

        if not record:
            record = PRUnderstanding(
                pull_request_id=pr_id,
                summary=understanding.summary,
                motivation_type=understanding.motivation.evidence_type.value,
                motivation_reason=understanding.motivation.reason,
                motivation_quote=understanding.motivation.evidence_quote,
                components=understanding.components,
                change_types=understanding.change_types,
                impact=understanding.impact,
                architectural_change=understanding.architectural_change,
                breaking_change=understanding.breaking_change,
                key_technical_details=understanding.key_technical_details,
                model_used=model_name,
            )
            self.session.add(record)
        else:
            record.summary = understanding.summary
            record.motivation_type = understanding.motivation.evidence_type.value
            record.motivation_reason = understanding.motivation.reason
            record.motivation_quote = understanding.motivation.evidence_quote
            record.components = understanding.components
            record.change_types = understanding.change_types
            record.impact = understanding.impact
            record.architectural_change = understanding.architectural_change
            record.breaking_change = understanding.breaking_change
            record.key_technical_details = understanding.key_technical_details
            record.model_used = model_name

        await self.session.flush()
        return record

    async def process_batch(
        self, request: PRUnderstandingProcessRequest
    ) -> PRUnderstandingProcessResponse:
        """Batch process unanalyzed (or forced) PRs for a repository."""
        # Find repository
        repo_stmt = select(Repository)
        if request.repository_id:
            repo_stmt = repo_stmt.where(Repository.id == request.repository_id)
        elif request.owner and request.repo:
            repo_stmt = repo_stmt.where(
                Repository.owner == request.owner, Repository.name == request.repo
            )
        else:
            raise ValueError("Must provide either repository_id or (owner, repo).")

        repo_res = await self.session.execute(repo_stmt)
        repo = repo_res.scalar_one_or_none()
        if not repo:
            raise ValueError("Repository not found.")

        # Find target PRs
        query = (
            select(PullRequest)
            .where(PullRequest.repository_id == repo.id)
            .options(
                selectinload(PullRequest.commits),
                selectinload(PullRequest.changed_files),
                selectinload(PullRequest.reviews),
                selectinload(PullRequest.review_comments),
                selectinload(PullRequest.understanding),
            )
            .order_by(PullRequest.number.asc())
        )

        if request.pr_number:
            query = query.where(PullRequest.number == request.pr_number)

        if not request.force_reprocess:
            query = query.outerjoin(
                PRUnderstanding, PullRequest.id == PRUnderstanding.pull_request_id
            ).where(PRUnderstanding.id.is_(None))

        if request.limit:
            query = query.limit(request.limit)

        prs = (await self.session.execute(query)).scalars().all()
        status_label = "all" if request.force_reprocess else "unanalyzed"
        logger.info(
            "Found %d %s PRs to analyze for %s/%s (force=%s).",
            len(prs),
            status_label,
            repo.owner,
            repo.name,
            request.force_reprocess,
        )

        processed_count = 0
        failed_count = 0

        for pr in prs:
            try:
                logger.info("Generating understanding for PR #%s: %s", pr.number, pr.title)
                result = await self.analyze_pull_request(pr)
                await self.save_understanding(
                    pr_id=pr.id,
                    understanding=result,
                    model_name=self.llm_service.model,
                )
                await self.session.commit()
                processed_count += 1
            except Exception as e:
                logger.exception("Failed to analyze PR #%s: %s", pr.number, e)
                await self.session.rollback()
                failed_count += 1

        return PRUnderstandingProcessResponse(
            repository=f"{repo.owner}/{repo.name}",
            status="completed",
            processed_count=processed_count,
            failed_count=failed_count,
            message=f"Analyzed {processed_count} PRs successfully ({failed_count} failed).",
        )
