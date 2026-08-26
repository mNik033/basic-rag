import React, { useState, useEffect } from 'react';
import { Database, RefreshCw, Layers, CheckCircle, PlusCircle, AlertCircle } from 'lucide-react';

export default function RepoManager({ repositories, onRefresh, selectedRepo, onSelectRepo }) {
  const [owner, setOwner] = useState('');
  const [repo, setRepo] = useState('');
  const [limit, setLimit] = useState(50);
  const [isSyncing, setIsSyncing] = useState(false);
  const [syncStatus, setSyncStatus] = useState('');
  const [isIndexing, setIsIndexing] = useState(false);
  const [indexStatus, setIndexStatus] = useState('');
  const [knowledgeStats, setKnowledgeStats] = useState({});

  const fetchKnowledgeStatus = async (repoFullName) => {
    if (!repoFullName) return;
    const [rOwner, rRepo] = repoFullName.split('/');
    try {
      const res = await fetch(`/api/v1/github/knowledge/status/${rOwner}/${rRepo}`);
      if (res.ok) {
        const data = await res.json();
        setKnowledgeStats((prev) => ({ ...prev, [repoFullName]: data }));
      }
    } catch (err) {
      console.error('Failed to fetch knowledge status:', err);
    }
  };

  useEffect(() => {
    repositories.forEach((r) => fetchKnowledgeStatus(r.full_name));
  }, [repositories]);

  const handleSyncRepo = async (e) => {
    e.preventDefault();
    if (!owner.trim() || !repo.trim()) return;

    setIsSyncing(true);
    setSyncStatus(`Syncing historical PRs for ${owner}/${repo}...`);

    try {
      const res = await fetch('/api/v1/github/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          owner: owner.trim(),
          repo: repo.trim(),
          max_prs: Number(limit),
        }),
      });

      if (res.ok) {
        const data = await res.json();
        setSyncStatus(`Sync completed: ${data.prs_synced} PRs synced.`);
        onRefresh();
        onSelectRepo(`${owner.trim()}/${repo.trim()}`);
      } else {
        setSyncStatus(`Sync failed with status ${res.status}`);
      }
    } catch (err) {
      setSyncStatus(`Error: ${err.message}`);
    } finally {
      setIsSyncing(false);
    }
  };

  const handleIndexKnowledge = async (repoFullName) => {
    if (!repoFullName) return;
    const [rOwner, rRepo] = repoFullName.split('/');
    setIsIndexing(true);
    setIndexStatus(`Generating vector embeddings for ${repoFullName}...`);

    try {
      const res = await fetch('/api/v1/github/knowledge/index', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          owner: rOwner,
          repo: rRepo,
          limit: 100,
          force_reindex: false,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        setIndexStatus(`Knowledge indexed: ${data.documents_indexed} docs embedded.`);
        await fetchKnowledgeStatus(repoFullName);
      } else {
        setIndexStatus('Indexing failed.');
      }
    } catch (err) {
      setIndexStatus(`Error: ${err.message}`);
    } finally {
      setIsIndexing(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col lg:flex-row gap-4 h-full min-h-0 overflow-hidden">
      {/* Ingest Repository Form */}
      <div className="w-full lg:w-96 flex flex-col bg-[#0d1322] border border-slate-800 rounded-lg p-5">
        <div className="flex items-center space-x-2 pb-3 border-b border-slate-800 mb-4">
          <PlusCircle className="w-4 h-4 text-sky-400" />
          <h2 className="text-sm font-semibold font-mono text-slate-100">Ingest Repository</h2>
        </div>

        <form onSubmit={handleSyncRepo} className="space-y-3.5 text-xs font-mono">
          <div>
            <label className="text-slate-400 block mb-1">GitHub Owner / Org</label>
            <input
              type="text"
              placeholder="e.g. facebook"
              value={owner}
              onChange={(e) => setOwner(e.target.value)}
              required
              className="w-full bg-[#090d16] border border-slate-800 rounded-md px-3 py-2 text-slate-200 focus:outline-none focus:border-sky-500"
            />
          </div>

          <div>
            <label className="text-slate-400 block mb-1">Repository Name</label>
            <input
              type="text"
              placeholder="e.g. react"
              value={repo}
              onChange={(e) => setRepo(e.target.value)}
              required
              className="w-full bg-[#090d16] border border-slate-800 rounded-md px-3 py-2 text-slate-200 focus:outline-none focus:border-sky-500"
            />
          </div>

          <div>
            <label className="text-slate-400 block mb-1">Max PRs to Ingest</label>
            <input
              type="number"
              min="1"
              max="500"
              value={limit}
              onChange={(e) => setLimit(e.target.value)}
              className="w-full bg-[#090d16] border border-slate-800 rounded-md px-3 py-2 text-slate-200 focus:outline-none focus:border-sky-500"
            />
          </div>

          <button
            type="submit"
            disabled={isSyncing}
            className="w-full bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white py-2 rounded-md font-semibold transition-colors flex items-center justify-center space-x-1.5 mt-2"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isSyncing ? 'animate-spin' : ''}`} />
            <span>{isSyncing ? 'Ingesting PRs...' : 'Trigger Sync'}</span>
          </button>

          {syncStatus && (
            <div className="p-2.5 bg-slate-900 border border-slate-800 rounded text-slate-300 text-[11px] font-mono">
              {syncStatus}
            </div>
          )}
        </form>
      </div>

      {/* Tracked Repositories List */}
      <div className="flex-1 flex flex-col bg-[#0d1322] border border-slate-800 rounded-lg p-5 overflow-hidden">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-4">
          <div className="flex items-center space-x-2">
            <Database className="w-4 h-4 text-sky-400" />
            <h2 className="text-sm font-semibold font-mono text-slate-100">Tracked Repositories</h2>
          </div>
          <button
            onClick={onRefresh}
            className="text-xs font-mono text-slate-400 hover:text-slate-200 flex items-center space-x-1"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Refresh</span>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto space-y-3 pr-1">
          {repositories.length === 0 ? (
            <div className="text-center py-16 text-slate-600 font-mono text-xs">
              No repositories synced yet. Use the form on the left to sync your first repository.
            </div>
          ) : (
            repositories.map((r) => {
              const stats = knowledgeStats[r.full_name] || {};
              return (
                <div
                  key={r.id}
                  className={`p-4 rounded-lg border text-xs font-mono transition-colors ${
                    selectedRepo === r.full_name
                      ? 'bg-slate-900 border-sky-500'
                      : 'bg-slate-900/60 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="text-sm font-bold text-slate-100 font-sans">
                      {r.full_name}
                    </div>
                    <button
                      onClick={() => onSelectRepo(r.full_name)}
                      className="text-sky-400 hover:underline text-[11px]"
                    >
                      {selectedRepo === r.full_name ? 'Selected' : 'Select Scope'}
                    </button>
                  </div>

                  <div className="grid grid-cols-3 gap-2 bg-slate-950/80 p-2.5 rounded border border-slate-800/80 mb-3 text-[11px]">
                    <div>
                      <span className="text-slate-500 block">Total PRs:</span>
                      <span className="text-slate-200 font-bold">{r.total_prs}</span>
                    </div>
                    <div>
                      <span className="text-slate-500 block">Analyzed (LLM):</span>
                      <span className="text-sky-400 font-bold">{stats.understood_prs ?? '0'}</span>
                    </div>
                    <div>
                      <span className="text-slate-500 block">Vector Indexed:</span>
                      <span className="text-emerald-400 font-bold">{stats.indexed_vectors ?? '0'}</span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between pt-1">
                    <span className="text-slate-500 text-[11px]">Branch: {r.default_branch}</span>
                    <button
                      onClick={() => handleIndexKnowledge(r.full_name)}
                      disabled={isIndexing}
                      className="bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-200 px-3 py-1 rounded text-xs transition-colors flex items-center space-x-1"
                    >
                      <Layers className="w-3 h-3 text-sky-400" />
                      <span>Index into Vector Store</span>
                    </button>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {indexStatus && (
          <div className="mt-3 p-2 bg-slate-900 border border-slate-800 rounded text-slate-300 text-xs font-mono text-center">
            {indexStatus}
          </div>
        )}
      </div>
    </div>
  );
}
