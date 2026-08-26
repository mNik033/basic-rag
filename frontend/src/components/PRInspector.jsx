import React, { useState, useEffect } from 'react';
import { GitPullRequest, Search, CheckCircle, AlertCircle, RefreshCw, Cpu, Layers } from 'lucide-react';

export default function PRInspector({ selectedRepo, targetPrNumber, onRefreshRepos }) {
  const [prs, setPrs] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedPr, setSelectedPr] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processStatus, setProcessStatus] = useState('');

  const fetchPRs = async () => {
    if (!selectedRepo) return;
    setIsLoading(true);
    const [owner, repo] = selectedRepo.split('/');
    try {
      const res = await fetch(`/api/v1/github/understanding/${owner}/${repo}?limit=50`);
      if (res.ok) {
        const data = await res.json();
        setPrs(data);
        if (data.length > 0) {
          if (targetPrNumber) {
            const matched = data.find((p) => p.pr_number === targetPrNumber);
            setSelectedPr(matched || data[0]);
          } else if (!selectedPr) {
            setSelectedPr(data[0]);
          }
        }
      }
    } catch (err) {
      console.error('Failed to load PR understandings:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchPRs();
  }, [selectedRepo, targetPrNumber]);

  const handleRunUnderstanding = async () => {
    if (!selectedRepo) return;
    const [owner, repo] = selectedRepo.split('/');
    setIsProcessing(true);
    setProcessStatus('Analyzing un-summarized PRs with LLM...');

    try {
      const res = await fetch('/api/v1/github/understanding/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          owner,
          repo,
          limit: 10,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setProcessStatus(`Processed ${data.processed_count} PRs successfully.`);
        await fetchPRs();
        if (onRefreshRepos) onRefreshRepos();
      } else {
        setProcessStatus('PR analysis failed.');
      }
    } catch (err) {
      setProcessStatus(`Error: ${err.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex-1 flex gap-4 h-full min-h-0 overflow-hidden">
      {/* Left List of PRs */}
      <div className="w-80 flex flex-col bg-[#0d1322] border border-slate-800 rounded-lg p-3 overflow-hidden">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800 mb-2">
          <span className="text-xs font-mono font-semibold text-slate-300">
            Analyzed PRs ({prs.length})
          </span>
          <button
            onClick={fetchPRs}
            disabled={isLoading}
            className="p-1 text-slate-400 hover:text-slate-200 transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto space-y-1.5 pr-1">
          {prs.length === 0 ? (
            <div className="text-xs text-slate-600 font-mono text-center py-8">
              No analyzed PRs found for {selectedRepo || 'this repository'}.
            </div>
          ) : (
            prs.map((p) => (
              <div
                key={p.pr_number}
                onClick={() => setSelectedPr(p)}
                className={`p-2.5 rounded border text-xs cursor-pointer transition-colors ${
                  selectedPr?.pr_number === p.pr_number
                    ? 'bg-slate-900 border-sky-500'
                    : 'bg-slate-900/50 border-slate-800 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-mono text-sky-400 font-bold">PR #{p.pr_number}</span>
                  <span className={`text-[10px] font-mono px-1 py-0.2 rounded uppercase ${
                    p.motivation?.evidence_type === 'documented'
                      ? 'bg-emerald-950 text-emerald-300'
                      : 'bg-amber-950 text-amber-300'
                  }`}>
                    {p.motivation?.evidence_type || 'Unknown'}
                  </span>
                </div>
                <div className="text-slate-200 font-sans truncate text-xs">{p.pr_title}</div>
              </div>
            ))
          )}
        </div>

        {/* Trigger analysis button */}
        <div className="pt-2 border-t border-slate-800 mt-2">
          <button
            onClick={handleRunUnderstanding}
            disabled={isProcessing || !selectedRepo}
            className="w-full bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-200 text-xs font-mono py-1.5 rounded flex items-center justify-center space-x-1.5 transition-colors"
          >
            <Cpu className="w-3.5 h-3.5 text-sky-400" />
            <span>{isProcessing ? 'Analyzing...' : 'Run PR Understanding'}</span>
          </button>
          {processStatus && (
            <div className="text-[10px] text-slate-400 font-mono text-center mt-1 truncate">
              {processStatus}
            </div>
          )}
        </div>
      </div>

      {/* Right Detail Pane */}
      <div className="flex-1 flex flex-col bg-[#0d1322] border border-slate-800 rounded-lg p-5 overflow-y-auto">
        {selectedPr ? (
          <div className="space-y-4">
            <div className="border-b border-slate-800 pb-3">
              <div className="flex items-center space-x-2 text-xs font-mono text-sky-400 mb-1">
                <GitPullRequest className="w-4 h-4" />
                <span className="font-bold text-sm">PR #{selectedPr.pr_number}</span>
                <span className="text-slate-500">•</span>
                <span className="text-slate-400">Author: {selectedPr.author}</span>
              </div>
              <h2 className="text-base font-semibold text-slate-100 font-sans">
                {selectedPr.pr_title}
              </h2>
            </div>

            {/* Summary Box */}
            <div className="bg-slate-900/80 p-3.5 rounded-lg border border-slate-800">
              <h3 className="text-xs font-mono uppercase text-slate-400 tracking-wider mb-1.5">
                Engineering Summary
              </h3>
              <p className="text-sm text-slate-200 font-sans leading-relaxed">
                {selectedPr.summary}
              </p>
            </div>

            {/* Motivation Box */}
            <div className="bg-slate-900/80 p-3.5 rounded-lg border border-slate-800">
              <div className="flex items-center justify-between mb-1.5">
                <h3 className="text-xs font-mono uppercase text-slate-400 tracking-wider">
                  Motivation & Problem
                </h3>
                <span className={`text-[10px] font-mono px-2 py-0.5 rounded uppercase border ${
                  selectedPr.motivation?.evidence_type === 'documented'
                    ? 'bg-emerald-950 text-emerald-300 border-emerald-800'
                    : 'bg-amber-950 text-amber-300 border-amber-800'
                }`}>
                  Evidence: {selectedPr.motivation?.evidence_type}
                </span>
              </div>
              <p className="text-sm text-slate-200 font-sans leading-relaxed">
                {selectedPr.motivation?.reason}
              </p>
              {selectedPr.motivation?.evidence_quote && (
                <div className="mt-2 text-xs font-sans italic text-slate-400 border-l-2 border-slate-700 pl-2.5">
                  "{selectedPr.motivation?.evidence_quote}"
                </div>
              )}
            </div>

            {/* Metadata Badges */}
            <div className="grid grid-cols-2 gap-3 text-xs font-mono">
              <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
                <span className="text-slate-500 block mb-1">Impacted Components:</span>
                <div className="flex flex-wrap gap-1">
                  {selectedPr.components?.map((c, i) => (
                    <span key={i} className="bg-slate-800 text-sky-300 px-2 py-0.5 rounded text-xs">
                      {c}
                    </span>
                  )) || 'None'}
                </div>
              </div>

              <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
                <span className="text-slate-500 block mb-1">Change Categories:</span>
                <div className="flex flex-wrap gap-1">
                  {selectedPr.change_types?.map((ct, i) => (
                    <span key={i} className="bg-slate-800 text-slate-200 px-2 py-0.5 rounded text-xs">
                      {ct}
                    </span>
                  )) || 'None'}
                </div>
              </div>
            </div>

            {/* Technical Details */}
            {selectedPr.key_technical_details?.length > 0 && (
              <div className="bg-slate-900/60 p-3.5 rounded-lg border border-slate-800 text-xs font-mono">
                <span className="text-slate-500 block mb-1.5 font-bold uppercase tracking-wider">
                  Technical Details & Data Structures:
                </span>
                <ul className="list-disc list-inside space-y-1 text-slate-300">
                  {selectedPr.key_technical_details.map((t, idx) => (
                    <li key={idx}>{t}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-24 text-slate-600 font-mono text-xs">
            Select a Pull Request on the left to inspect its structured intelligence.
          </div>
        )}
      </div>
    </div>
  );
}
