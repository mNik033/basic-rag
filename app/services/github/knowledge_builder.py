from typing import Any, Dict, List, Optional
from app.domain.knowledge import EngineeringDocument
from app.domain.models import ChangedFile, Commit, PRUnderstanding, PullRequest, Repository


class EngineeringDocumentSynthesizer:
    """Synthesizes structured engineering documents and metadata from Pull Requests and their AI understandings."""

    @staticmethod
    def synthesize_pr_document(
        repo: Repository,
        pr: PullRequest,
        understanding: Optional[PRUnderstanding] = None,
    ) -> EngineeringDocument:
        """Create a single rich composite document optimized for dense vector retrieval."""
        doc_id = f"gh_pr_{repo.owner}_{repo.name}_{pr.number}"
        repo_full_name = f"{repo.owner}/{repo.name}"

        # 1. Construct Document Text
        sections: List[str] = [
            f"# Repository: {repo_full_name} | PR #{pr.number}: {pr.title}",
            f"Author: {pr.author} | State: {pr.state} | Merged: {pr.merged_at.isoformat() if pr.merged_at else 'N/A'}",
            f"Milestone / Release: {pr.milestone or 'N/A'}",
        ]

        if pr.labels:
            sections.append(f"Labels: {', '.join(pr.labels)}")

        # AI Understanding Section
        if understanding:
            sections.append("\n## Engineering Summary:")
            sections.append(understanding.summary)

            sections.append(f"\n## Motivation ({understanding.motivation_type.capitalize()}):")
            sections.append(understanding.motivation_reason)
            if understanding.motivation_quote:
                sections.append(f"> \"{understanding.motivation_quote}\"")

            if understanding.components:
                sections.append(f"\nImpacted Components: {', '.join(understanding.components)}")

            if understanding.change_types:
                sections.append(f"Change Categories: {', '.join(understanding.change_types)}")

            if understanding.impact:
                sections.append("Impact:")
                for imp in understanding.impact:
                    sections.append(f"- {imp}")

            sections.append(f"Architectural Change: {'Yes' if understanding.architectural_change else 'No'}")
            sections.append(f"Breaking Change: {'Yes' if understanding.breaking_change else 'No'}")

            if understanding.key_technical_details:
                sections.append("Key Technical Details:")
                for tech in understanding.key_technical_details:
                    sections.append(f"- {tech}")

        # Original PR Description
        if pr.body and pr.body.strip():
            sections.append("\n## Original PR Description:")
            sections.append(pr.body.strip()[:1500])

        # Commit messages
        if pr.commits:
            sections.append("\n## Commits:")
            for c in pr.commits[:8]:
                sections.append(f"- [{c.sha[:7]}] {c.message.strip()}")

        # Changed Files
        if pr.changed_files:
            sections.append("\n## Changed Files:")
            for f in pr.changed_files[:15]:
                sections.append(f"- {f.filename} ({f.status}, +{f.additions}/-{f.deletions})")

        document_text = "\n".join(sections)

        # 2. Construct Searchable Metadata (used for hybrid keyword + scalar filtering)
        metadata: Dict[str, Any] = {
            "doc_type": "github_pr",
            "repository": repo_full_name,
            "owner": repo.owner,
            "repo": repo.name,
            "pr_number": pr.number,
            "pr_title": pr.title,
            "author": pr.author,
            "state": pr.state,
            "merged_at": pr.merged_at.isoformat() if pr.merged_at else "",
            "milestone": pr.milestone or "",
            "labels": ",".join(pr.labels) if pr.labels else "",
            "components": ",".join(understanding.components) if (understanding and understanding.components) else "",
            "change_types": ",".join(understanding.change_types) if (understanding and understanding.change_types) else "",
            "architectural_change": bool(understanding.architectural_change) if understanding else False,
            "breaking_change": bool(understanding.breaking_change) if understanding else False,
            "has_ai_understanding": understanding is not None,
        }

        return EngineeringDocument(
            doc_id=doc_id,
            text=document_text,
            metadata=metadata,
        )
