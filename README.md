# Engineering Memory

Search historical GitHub activity across pull requests, diffs, commits, and review comments. Answers questions with direct citations to past PRs.

## Quick start

### 1. Prerequisites
- Python 3.13+
- Node.js 18+ and npm
- Ollama running locally with a model installed (for example `ollama run gemma4:e2b`)
- Optional: a GitHub Personal Access Token for private repositories or higher rate limits

### 2. Backend setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment (optional: set GITHUB_TOKEN)
cp .env.example .env

# 3. Start the FastAPI server
uvicorn app.main:create_application --factory --reload --port 8000
```

The backend runs on `http://localhost:8000`. API documentation is available at `http://localhost:8000/api/v1/docs`.

### 3. Frontend setup
```bash
# In a separate terminal:
cd frontend
npm install
npm run dev
```

The frontend dashboard opens at `http://localhost:5173`.

## CLI commands

You can also run all tasks from the command line:

```bash
# Fetch pull requests from a repository
python scripts/sync_github.py --owner torvalds --repo linux --limit 50

# Extract summaries and motivation from pull requests
python scripts/analyze_prs.py --owner torvalds --repo linux --limit 50

# Build vector embeddings and index them into ChromaDB
python scripts/index_knowledge.py --owner torvalds --repo linux

# Ask engineering questions with citations
python scripts/ask_engineering.py "Which PRs optimized memory management?" --owner torvalds --repo linux --stream
```

## Tests

Run the test suite with pytest:
```bash
pytest
```

