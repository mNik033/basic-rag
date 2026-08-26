import React, { useState, useRef } from 'react';
import { Send, Filter, FileText, ArrowRight, Square } from 'lucide-react';
import MarkdownRenderer from './MarkdownRenderer';

export default function QAAssistant({ selectedRepo, onSelectPR }) {
  const [query, setQuery] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [answer, setAnswer] = useState('');
  const [scenario, setScenario] = useState(null);
  const [evidence, setEvidence] = useState([]);
  const [showFilters, setShowFilters] = useState(false);
  const [selectedEvidence, setSelectedEvidence] = useState(null);

  const abortControllerRef = useRef(null);

  // Filters state
  const [filters, setFilters] = useState({
    component: '',
    change_type: '',
    architectural_only: false,
    breaking_only: false,
  });

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
  };

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!query.trim() || isStreaming) return;

    setIsStreaming(true);
    setAnswer('');
    setEvidence([]);
    setScenario(null);
    setSelectedEvidence(null);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    const payload = {
      query: query.trim(),
      repository: selectedRepo || null,
      filter: {
        component: filters.component || null,
        change_type: filters.change_type || null,
        architectural_only: filters.architectural_only,
        breaking_only: filters.breaking_only,
      },
      limit: 5,
    };

    try {
      const response = await fetch('/api/v1/github/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let currentEvent = null;

        for (const line of lines) {
          if (line.startsWith('event:')) {
            currentEvent = line.replace('event:', '').trim();
          } else if (line.startsWith('data:')) {
            const rawData = line.replace('data:', '').trim();
            if (!rawData) continue;

            try {
              const data = JSON.parse(rawData);
              if (currentEvent === 'metadata') {
                setScenario(data.scenario);
                setEvidence(data.evidence || []);
              } else if (currentEvent === 'token') {
                setAnswer((prev) => prev + (data.token || ''));
              }
            } catch (err) {
              console.error('Error parsing SSE event data:', err);
            }
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        setAnswer(`Error connecting to engineering memory engine: ${err.message}`);
      }
    } finally {
      setIsStreaming(false);
      abortControllerRef.current = null;
    }
  };

  return (
    <div className="flex-1 flex flex-col lg:flex-row gap-4 h-full min-h-0 overflow-hidden">
      {/* Left / Main Q&A Column */}
      <div className="flex-1 flex flex-col bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-slate-800 rounded-lg p-4 overflow-hidden shadow-xs">
        {/* Top: Answer Stream Output Area */}
        <div className="flex-1 overflow-y-auto pb-4 pr-1">
          {scenario && (
            <div className="mb-3 flex items-center space-x-2">
              <span className="text-[11px] font-mono uppercase bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 px-2 py-0.5 rounded border border-slate-200 dark:border-slate-700">
                Intent: {scenario.replace('_', ' ')}
              </span>
              <span className="text-[11px] font-mono text-slate-500">
                {evidence.length} pull requests considered
              </span>
            </div>
          )}

          {answer ? (
            <div className="text-slate-800 dark:text-slate-200">
              <MarkdownRenderer
                content={answer}
                onSelectPR={(prNum) => onSelectPR && onSelectPR(prNum, selectedRepo)}
              />
            </div>
          ) : isStreaming ? (
            <div className="flex items-center space-x-2 text-xs font-mono text-slate-500 dark:text-slate-400 animate-pulse py-8">
              <span className="w-2 h-2 rounded-full bg-sky-500 dark:bg-sky-400"></span>
              <span>Searching indexed pull requests and synthesizing response...</span>
            </div>
          ) : (
            <div className="py-12 max-w-xl mx-auto space-y-4">
              <div className="text-xs font-mono text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                Ask about historical engineering changes:
              </div>

              <div className="grid grid-cols-1 gap-2.5 text-xs font-mono">
                <div
                  onClick={() => setQuery("What changed between release 5.2 and 5.3?")}
                  className="p-3 bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700 rounded-lg cursor-pointer transition-colors"
                >
                  <div className="text-slate-800 dark:text-slate-300 font-semibold mb-0.5">Release comparison</div>
                  <div className="text-slate-500 font-sans text-[11px]">"What changed between release 5.2 and 5.3?"</div>
                </div>

                <div
                  onClick={() => setQuery("Have we seen ImageCache memory issues before?")}
                  className="p-3 bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700 rounded-lg cursor-pointer transition-colors"
                >
                  <div className="text-slate-800 dark:text-slate-300 font-semibold mb-0.5">Historical bug or regression</div>
                  <div className="text-slate-500 font-sans text-[11px]">"Have we seen ImageCache memory issues before?"</div>
                </div>

                <div
                  onClick={() => setQuery("Which PRs affected performance or memory allocation?")}
                  className="p-3 bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700 rounded-lg cursor-pointer transition-colors"
                >
                  <div className="text-slate-800 dark:text-slate-300 font-semibold mb-0.5">Component & impact analysis</div>
                  <div className="text-slate-500 font-sans text-[11px]">"Which PRs affected performance or memory allocation?"</div>
                </div>

                <div
                  onClick={() => setQuery("Why was the scheduler architecture rewritten?")}
                  className="p-3 bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700 rounded-lg cursor-pointer transition-colors"
                >
                  <div className="text-slate-800 dark:text-slate-300 font-semibold mb-0.5">Architectural decisions</div>
                  <div className="text-slate-500 font-sans text-[11px]">"Why was the scheduler architecture rewritten?"</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Bottom Pinned: Filters Drawer & Chat Input Box */}
        <div className="pt-3 border-t border-slate-200 dark:border-slate-800/80 space-y-2">
          {/* Collapsible Metadata Filters */}
          {showFilters && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 p-3 bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800/80 rounded-lg text-xs font-mono mb-2">
              <div>
                <label className="text-slate-600 dark:text-slate-400 block mb-1">Component</label>
                <input
                  type="text"
                  placeholder="e.g. ImageCache"
                  value={filters.component}
                  disabled={isStreaming}
                  onChange={(e) => setFilters({ ...filters, component: e.target.value })}
                  className="w-full bg-white dark:bg-[#090d16] border border-slate-200 dark:border-slate-800 rounded px-2 py-1 text-slate-800 dark:text-slate-200 focus:outline-none focus:border-sky-500 disabled:opacity-50 disabled:cursor-not-allowed"
                />
              </div>
              <div>
                <label className="text-slate-600 dark:text-slate-400 block mb-1">Change Type</label>
                <input
                  type="text"
                  placeholder="e.g. memory, performance"
                  value={filters.change_type}
                  disabled={isStreaming}
                  onChange={(e) => setFilters({ ...filters, change_type: e.target.value })}
                  className="w-full bg-white dark:bg-[#090d16] border border-slate-200 dark:border-slate-800 rounded px-2 py-1 text-slate-800 dark:text-slate-200 focus:outline-none focus:border-sky-500 disabled:opacity-50 disabled:cursor-not-allowed"
                />
              </div>
              <div className="flex items-center space-x-2 pt-4">
                <input
                  type="checkbox"
                  id="arch_only"
                  checked={filters.architectural_only}
                  disabled={isStreaming}
                  onChange={(e) => setFilters({ ...filters, architectural_only: e.target.checked })}
                  className="rounded border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-sky-600 focus:ring-0 disabled:opacity-50 disabled:cursor-not-allowed"
                />
                <label htmlFor="arch_only" className={`text-slate-700 dark:text-slate-300 ${isStreaming ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}`}>Architectural only</label>
              </div>
              <div className="flex items-center space-x-2 pt-4">
                <input
                  type="checkbox"
                  id="break_only"
                  checked={filters.breaking_only}
                  disabled={isStreaming}
                  onChange={(e) => setFilters({ ...filters, breaking_only: e.target.checked })}
                  className="rounded border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-sky-600 focus:ring-0 disabled:opacity-50 disabled:cursor-not-allowed"
                />
                <label htmlFor="break_only" className={`text-slate-700 dark:text-slate-300 ${isStreaming ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}`}>Breaking only</label>
              </div>
            </div>
          )}

          {/* Chat Input Container */}
          <form onSubmit={handleSubmit} className="relative">
            <div className={`search-field-wrap p-[1px] rounded-lg ${isStreaming ? 'is-loading' : 'bg-slate-200 dark:bg-slate-700/80 focus-within:bg-sky-500'}`}>
              <div className="relative rounded-[7px] bg-white dark:bg-[#090d16] overflow-hidden flex items-center">
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  disabled={isStreaming}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && !isStreaming) {
                      e.preventDefault();
                      handleSubmit();
                    }
                  }}
                  placeholder={isStreaming ? 'Generating response...' : 'Ask an engineering question...'}
                  rows={2}
                  className="w-full bg-transparent border-0 rounded-[7px] pl-3.5 pr-24 py-2.5 text-sm text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none font-mono resize-none transition-all disabled:opacity-40 disabled:text-slate-400 disabled:cursor-not-allowed"
                />

                <div className="absolute right-2.5 bottom-2.5 flex items-center space-x-2">
                  <button
                    type="button"
                    onClick={() => setShowFilters(!showFilters)}
                    disabled={isStreaming}
                    title="Toggle Metadata Filters"
                    className={`p-1.5 rounded border text-xs font-mono transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                      showFilters || filters.component || filters.change_type || filters.architectural_only
                        ? 'bg-sky-100 dark:bg-sky-950/60 border-sky-300 dark:border-sky-800 text-sky-700 dark:text-sky-300'
                        : 'bg-slate-100 dark:bg-slate-900 border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                    }`}
                  >
                    <Filter className="w-3.5 h-3.5" />
                  </button>

                  {isStreaming ? (
                    <button
                      type="button"
                      onClick={handleStop}
                      className="bg-rose-100 dark:bg-rose-900/80 hover:bg-rose-200 dark:hover:bg-rose-800 text-rose-700 dark:text-rose-200 border border-rose-300 dark:border-rose-700/70 text-xs font-semibold px-3 py-1.5 rounded-md flex items-center space-x-1.5 transition-colors cursor-pointer"
                    >
                      <Square className="w-3 h-3 fill-current" />
                      <span>Stop</span>
                    </button>
                  ) : (
                    <button
                      type="submit"
                      disabled={!query.trim()}
                      className="bg-sky-600 hover:bg-sky-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-xs font-semibold px-3 py-1.5 rounded-md flex items-center space-x-1.5 transition-colors"
                    >
                      <span>Ask</span>
                      <Send className="w-3 h-3" />
                    </button>
                  )}
                </div>
              </div>
            </div>
            <div className="flex justify-between items-center px-1 pt-1 text-[10px] text-slate-500 font-mono">
              <span>{selectedRepo ? `Scoped to ${selectedRepo}` : 'Searching across all repositories'}</span>
              <span>{isStreaming ? 'Streaming response from LLM...' : 'Ctrl+Enter to send'}</span>
            </div>
          </form>
        </div>
      </div>

      {/* Right Column: Evidence Panel */}
      <div className="w-full lg:w-96 flex flex-col bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-slate-800 rounded-lg p-4 overflow-hidden shadow-xs">
        <div className="flex items-center justify-between pb-2 border-b border-slate-200 dark:border-slate-800 mb-3">
          <div className="flex items-center space-x-1.5 text-xs font-mono font-semibold text-slate-800 dark:text-slate-200">
            <FileText className="w-3.5 h-3.5 text-sky-600 dark:text-sky-400" />
            <span>Related Pull Requests ({evidence.length})</span>
          </div>
          {isStreaming && evidence.length === 0 && (
            <span className="text-[10px] font-mono text-sky-600 dark:text-sky-400 animate-pulse">Searching repository...</span>
          )}
        </div>

        <div className="flex-1 overflow-y-auto space-y-2.5 pr-1">
          {evidence.length === 0 ? (
            isStreaming ? (
              <div className="space-y-2 py-2">
                {[1, 2, 3].map((n) => (
                  <div key={n} className="p-3 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/40 animate-pulse space-y-2">
                    <div className="h-3 bg-slate-200 dark:bg-slate-800 rounded w-1/3"></div>
                    <div className="h-3 bg-slate-200/60 dark:bg-slate-800/60 rounded w-5/6"></div>
                    <div className="h-2 bg-slate-200/40 dark:bg-slate-800/40 rounded w-1/2"></div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-xs text-slate-500 dark:text-slate-600 font-mono text-center py-12">
                No cited pull requests yet.
              </div>
            )
          ) : (
            evidence.map((item, idx) => (
              <div
                key={idx}
                onClick={() => setSelectedEvidence(item === selectedEvidence ? null : item)}
                className={`p-3 rounded-lg border text-xs font-mono transition-all cursor-pointer ${
                  selectedEvidence === item
                    ? 'bg-sky-50/70 dark:bg-slate-900 border-sky-400 dark:border-sky-500/80 shadow-xs'
                    : 'bg-slate-50 dark:bg-slate-900/60 border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center space-x-1.5">
                    <span className="bg-sky-100 dark:bg-sky-950 text-sky-800 dark:text-sky-300 border border-sky-300 dark:border-sky-800 px-1 py-0.2 rounded text-[10px] font-bold">
                      #{item.rank || idx + 1}
                    </span>
                    <span className="text-sky-700 dark:text-sky-400 font-semibold font-mono">
                      PR #{item.pr_number}
                    </span>
                  </div>
                  <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono uppercase ${
                    item.motivation_type === 'documented'
                      ? 'bg-emerald-100 dark:bg-emerald-950 text-emerald-800 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-800/60'
                      : 'bg-amber-100 dark:bg-amber-950 text-amber-800 dark:text-amber-300 border border-amber-300 dark:border-amber-800/60'
                  }`}>
                    {item.motivation_type || 'Unknown'}
                  </span>
                </div>

                <div className="text-slate-900 dark:text-slate-200 font-sans font-medium text-xs line-clamp-2 mb-1.5">
                  {item.title}
                </div>

                <div className="text-slate-600 dark:text-slate-400 text-[11px] space-y-1">
                  {item.motivation_reason && (
                    <div className="text-slate-700 dark:text-slate-300 font-sans italic line-clamp-2">
                      "{item.motivation_reason}"
                    </div>
                  )}

                  {item.components && item.components.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-1.5">
                      {item.components.map((c, ci) => (
                        <span key={ci} className="bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300 px-1.5 py-0.5 rounded text-[10px]">
                          {c}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                <div className="flex items-center justify-between mt-2 pt-2 border-t border-slate-200 dark:border-slate-800/80 text-[10px] text-slate-500">
                  <span>
                    {item.author} {item.milestone ? `• ${item.milestone}` : ''}
                  </span>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (onSelectPR) {
                        onSelectPR(item.pr_number, item.repository);
                      }
                    }}
                    className="text-sky-600 dark:text-sky-400 hover:underline flex items-center space-x-0.5 font-semibold"
                  >
                    <span>Inspect</span>
                    <ArrowRight className="w-2.5 h-2.5" />
                  </button>
                </div>

                {selectedEvidence === item && item.changed_files && item.changed_files.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-slate-200 dark:border-slate-800/80 text-[10px] text-slate-500 dark:text-slate-400 space-y-0.5">
                    <div className="text-slate-700 dark:text-slate-300 font-semibold mb-1">Changed Files:</div>
                    {item.changed_files.slice(0, 5).map((f, fi) => (
                      <div key={fi} className="truncate font-mono">
                        • {f}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

