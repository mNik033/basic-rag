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
Your role is to analyze historical GitHub Pull Requests (PR descriptions, commit messages, file diffs, reviews, and comments) to extract structured engineering knowledge.

CRITICAL INSTRUCTIONS ON MOTIVATION & EVIDENCE:
You must strictly classify the motivation for the change into one of three categories:
1. "documented": The PR description, commit messages, or review comments explicitly state the reason/goal (e.g. "Fixes memory leak in ImageCache under high load", "Reduces battery drain"). When documented, you MUST provide an exact short `evidence_quote`.
2. "inferred": The code diffs or commit messages strongly imply the reason, but it is NOT explicitly written down in documentation or comments.
3. "unknown": There is insufficient evidence in the PR or diffs to know why the change was made.

You must respond ONLY with a valid JSON object adhering strictly to this JSON schema:
{
  "summary": "Concise 1-2 sentence engineering summary of the change",
  "motivation": {
    "evidence_type": "documented" | "inferred" | "unknown",
    "reason": "Why the change was made (the problem solved or requirement met)",
    "evidence_quote": "Direct quote from PR description or review if documented, else null"
  },
  "components": ["List of impacted modules, classes, subsystems, or packages"],
  "change_types": ["List of categories: e.g. memory, performance, bugfix, refactor, feature, security, api-change, build-infra"],
  "impact": ["List of concrete technical and behavioral impacts"],
  "architectural_change": true | false,
  "breaking_change": true | false,
  "key_technical_details": ["Key technical highlights: data structures, algorithms, configs, APIs modified"]
}
Do not include any conversational preamble or markdown code fencing outside the JSON. Return only the JSON object.
"""


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
        self.llm_service = llm_service or get_llm_service()
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

        # Changed Files & Diffs
        if pr.changed_files:
            parts.append("## Changed Files & Patches:")
            current_diff_len = 0
            for f in pr.changed_files:
                diff_header = f"### File: {f.filename} ({f.status}, +{f.additions}/-{f.deletions})"
                patch = f.patch_text or "(Binary or diff too large)"

                # Truncate large individual patches
                if len(patch) > 2000:
                    patch = patch[:2000] + "\n... [diff truncated for length] ..."

                diff_block = f"{diff_header}\n```diff\n{patch}\n```\n"

                if current_diff_len + len(diff_block) > self.max_diff_chars:
                    parts.append(f"{diff_header} [Full diff omitted to fit context budget]")
                else:
                    parts.append(diff_block)
                    current_diff_len += len(diff_block)
            parts.append("")

        # Reviews & Discussion
        if pr.reviews or pr.review_comments:
            parts.append("## Review Comments & Discussion:")
            for r in pr.reviews:
                if r.body and r.body.strip():
                    parts.append(f"Review by @{r.author} ({r.state}): {r.body.strip()}")
            for rc in pr.review_comments:
                loc = f" on {rc.path}:{rc.line}" if rc.path else ""
                parts.append(f"Comment by @{rc.author}{loc}: {rc.body.strip()}")
            parts.append("")

        parts.append("Please analyze the above Pull Request and provide the structured JSON engineering summary.")
        return "\n".join(parts)

    async def analyze_pull_request(self, pr: PullRequest) -> PRUnderstandingResult:
        """Call LLM with PR context and parse into validated PRUnderstandingResult."""
        prompt = self.build_pr_context_prompt(pr)
        raw_response = await self.llm_service.generate_response(
            prompt=prompt,
            system_prompt=PR_UNDERSTANDING_SYSTEM_PROMPT,
            format_json=True,
        )

        cleaned = clean_json_response(raw_response)

        try:
            parsed_dict = json.loads(cleaned)
            result = PRUnderstandingResult.model_validate(parsed_dict)
            result.raw_response = parsed_dict
            return result
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Failed strict validation on LLM output for PR #%s: %s. Attempting fallback.", pr.number, exc)
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
            query = query.where(PullRequest.understanding == None)  # noqa: E711

        if request.limit:
            query = query.limit(request.limit)

        prs = (await self.session.execute(query)).scalars().all()

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
