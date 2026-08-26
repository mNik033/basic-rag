import React, { useState } from 'react';
import { Search, Sliders, ArrowUpDown, Filter, FileText } from 'lucide-react';

export default function HybridSearch({ selectedRepo }) {
  const [query, setQuery] = useState('');
  const [vectorWeight, setVectorWeight] = useState(0.6);
  const [keywordWeight, setKeywordWeight] = useState(0.4);
  const [limit, setLimit] = useState(10);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);

  // Filters
  const [component, setComponent] = useState('');
  const [changeType, setChangeType] = useState('');
  const [architecturalOnly, setArchitecturalOnly] = useState(false);
  const [breakingOnly, setBreakingOnly] = useState(false);

  const handleSearch = async (e) => {
    if (e) e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    try {
      const res = await fetch('/api/v1/github/search/hybrid', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query.trim(),
          filter: {
            repository: selectedRepo || null,
            component: component || null,
            change_type: changeType || null,
            architectural_only: architecturalOnly,
            breaking_only: breakingOnly,
          },
          limit: Number(limit),
          vector_weight: parseFloat(vectorWeight),
          keyword_weight: parseFloat(keywordWeight),
        }),
      });

      if (res.ok) {
        const data = await res.json();
        setResults(data);
      }
    } catch (err) {
      console.error('Hybrid search failed:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col bg-[#0d1322] border border-slate-800 rounded-lg p-4 overflow-hidden">
      {/* Search Header */}
      <form onSubmit={handleSearch} className="space-y-3 pb-3 border-b border-slate-800">
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="w-4 h-4 absolute left-3 top-3 text-slate-500" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search engineering history using Hybrid RRF (e.g. ImageCache eviction memory leak)..."
              className="w-full bg-[#090d16] border border-slate-700/80 rounded-lg pl-9 pr-3.5 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-sky-500 font-mono"
            />
          </div>
          <button
            type="submit"
            disabled={!query.trim() || isLoading}
            className="bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white text-xs font-semibold px-4 py-2 rounded-lg transition-colors font-mono"
          >
            {isLoading ? 'Searching...' : 'Search'}
          </button>
        </div>

        {/* Weights and Sliders */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 bg-slate-900/60 p-3 rounded-lg border border-slate-800/80 text-xs font-mono">
          <div>
            <div className="flex justify-between text-slate-400 mb-1">
              <span>Vector Weight:</span>
              <span className="text-sky-400 font-semibold">{vectorWeight}</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={vectorWeight}
              onChange={(e) => setVectorWeight(e.target.value)}
              className="w-full accent-sky-500 cursor-pointer"
            />
          </div>

          <div>
            <div className="flex justify-between text-slate-400 mb-1">
              <span>Keyword Weight:</span>
              <span className="text-sky-400 font-semibold">{keywordWeight}</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={keywordWeight}
              onChange={(e) => setKeywordWeight(e.target.value)}
              className="w-full accent-sky-500 cursor-pointer"
            />
          </div>

          <div>
            <label className="text-slate-400 block mb-1">Component</label>
            <input
              type="text"
              placeholder="e.g. Scheduler"
              value={component}
              onChange={(e) => setComponent(e.target.value)}
              className="w-full bg-[#090d16] border border-slate-800 rounded px-2 py-1 text-slate-200 focus:outline-none focus:border-sky-500"
            />
          </div>

          <div>
            <label className="text-slate-400 block mb-1">Change Type</label>
            <input
              type="text"
              placeholder="e.g. memory"
              value={changeType}
              onChange={(e) => setChangeType(e.target.value)}
              className="w-full bg-[#090d16] border border-slate-800 rounded px-2 py-1 text-slate-200 focus:outline-none focus:border-sky-500"
            />
          </div>
        </div>
      </form>

      {/* Results Container */}
      <div className="flex-1 overflow-y-auto mt-3">
        {results ? (
          <div>
            <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-3">
              <span>Found {results.total_candidates} candidates</span>
              <div className="flex space-x-3 text-[11px]">
                <span className="text-sky-400">Vector hits: {results.vector_hits}</span>
                <span className="text-emerald-400">Keyword hits: {results.keyword_hits}</span>
              </div>
            </div>

            <div className="space-y-2">
              {results.results.map((r) => (
                <div key={r.pr_number} className="p-3 bg-slate-900/70 border border-slate-800 rounded-lg text-xs font-mono hover:border-slate-700 transition-colors">
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center space-x-2">
                      <span className="bg-sky-950 text-sky-300 border border-sky-800 px-1.5 py-0.5 rounded font-semibold">
                        #{r.rank}
                      </span>
                      <span className="text-slate-200 font-sans font-semibold text-sm">
                        PR #{r.pr_number}: {r.title}
                      </span>
                    </div>
                    <div className="text-slate-400 text-[11px]">
                      RRF Score: <span className="text-sky-400 font-semibold">{r.combined_score}</span>
                    </div>
                  </div>

                  {r.summary && (
                    <div className="text-slate-300 font-sans text-xs my-1">
                      {r.summary}
                    </div>
                  )}

                  <div className="flex flex-wrap gap-2 text-[11px] text-slate-400 mt-2">
                    {r.author && <span>Author: {r.author}</span>}
                    {r.milestone && <span>Milestone: {r.milestone}</span>}
                    {r.components && r.components.length > 0 && (
                      <span className="text-sky-400">Components: {r.components.join(', ')}</span>
                    )}
                  </div>

                  {r.match_reasons && r.match_reasons.length > 0 && (
                    <div className="mt-2 text-[10px] text-slate-500 border-t border-slate-800/80 pt-1.5">
                      Match: {r.match_reasons.join(' | ')}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="text-center py-16 text-slate-600 font-mono text-xs">
            Run a search query to inspect hybrid rank fusion scores and candidate PR evidence.
          </div>
        )}
      </div>
    </div>
  );
}
