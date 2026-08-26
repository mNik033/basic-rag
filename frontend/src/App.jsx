import React, { useState, useEffect } from 'react';
import {
  MessageSquare,
  GitPullRequest,
  Database,
  Terminal,
  Github,
  Sun,
  Moon,
} from 'lucide-react';

import QAAssistant from './components/QAAssistant';
import RepoManager from './components/RepoManager';
import PRInspector from './components/PRInspector';

export default function App() {
  const [activeTab, setActiveTab] = useState('qa'); // 'qa' | 'prs' | 'repositories'
  const [selectedRepo, setSelectedRepo] = useState('');
  const [selectedPrNumber, setSelectedPrNumber] = useState(null);
  const [repositories, setRepositories] = useState([]);
  const [isLoadingRepos, setIsLoadingRepos] = useState(false);
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'dark');

  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('theme', theme);
  }, [theme]);

  // Fetch tracked repositories on load
  const fetchRepositories = async () => {
    setIsLoadingRepos(true);
    try {
      const res = await fetch('/api/v1/github/repositories');
      if (res.ok) {
        const data = await res.json();
        setRepositories(data);
        if (data.length > 0 && !selectedRepo) {
          setSelectedRepo(data[0].full_name);
        }
      }
    } catch (err) {
      console.error('Failed to load repositories:', err);
    } finally {
      setIsLoadingRepos(false);
    }
  };

  useEffect(() => {
    fetchRepositories();
  }, []);

  const handleSelectPRFromQA = (prNumber, repoFullName) => {
    if (repoFullName) {
      setSelectedRepo(repoFullName);
    }
    setSelectedPrNumber(prNumber);
    setActiveTab('prs');
  };

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  };

  return (
    <div className="flex flex-col min-h-screen bg-slate-50 dark:bg-[#090d16] text-slate-900 dark:text-slate-100 antialiased font-sans transition-colors">
      {/* Top Header */}
      <header className="h-14 border-b border-slate-200 dark:border-slate-800 bg-white/90 dark:bg-[#0d1322]/90 backdrop-blur px-4 flex items-center justify-between sticky top-0 z-50 relative">
        {/* Left: Branding & Repo Selector */}
        <div className="flex items-center space-x-3 z-10">
          <div className="flex items-center space-x-2 text-sky-600 dark:text-sky-400 font-semibold tracking-tight">
            <Terminal className="w-5 h-5" />
            <span className="text-slate-900 dark:text-slate-100 font-mono text-sm tracking-wider font-bold">ENGINEERING MEMORY</span>
          </div>
          <span className="text-slate-300 dark:text-slate-700">/</span>
          <div className="flex items-center space-x-1.5 text-xs text-slate-600 dark:text-slate-400 font-mono bg-slate-100 dark:bg-slate-900/80 px-2 py-1 rounded border border-slate-200 dark:border-slate-800">
            <Github className="w-3.5 h-3.5" />
            <select
              value={selectedRepo}
              onChange={(e) => setSelectedRepo(e.target.value)}
              className="bg-transparent text-slate-800 dark:text-slate-200 text-xs font-mono focus:outline-none cursor-pointer pr-1"
            >
              <option value="" className="bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-400">All Repositories</option>
              {repositories.map((r) => (
                <option key={r.id} value={r.full_name} className="bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200">
                  {r.full_name} ({r.total_prs} PRs)
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Center: Tabs Navigation (Strictly centered on the viewport) */}
        <nav className="hidden sm:flex items-center space-x-1 bg-slate-100 dark:bg-slate-900/90 p-1 rounded-lg border border-slate-200 dark:border-slate-800/80 absolute left-1/2 -translate-x-1/2">
          <button
            onClick={() => setActiveTab('qa')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              activeTab === 'qa'
                ? 'bg-white dark:bg-sky-950 text-sky-700 dark:text-sky-300 border border-slate-200 dark:border-sky-800/60 shadow-xs'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-200/60 dark:hover:bg-slate-800/60'
            }`}
          >
            <MessageSquare className="w-3.5 h-3.5" />
            <span>Q&A Assistant</span>
          </button>

          <button
            onClick={() => setActiveTab('prs')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              activeTab === 'prs'
                ? 'bg-white dark:bg-sky-950 text-sky-700 dark:text-sky-300 border border-slate-200 dark:border-sky-800/60 shadow-xs'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-200/60 dark:hover:bg-slate-800/60'
            }`}
          >
            <GitPullRequest className="w-3.5 h-3.5" />
            <span>PR Intelligence</span>
          </button>

          <button
            onClick={() => setActiveTab('repositories')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              activeTab === 'repositories'
                ? 'bg-white dark:bg-sky-950 text-sky-700 dark:text-sky-300 border border-slate-200 dark:border-sky-800/60 shadow-xs'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-200/60 dark:hover:bg-slate-800/60'
            }`}
          >
            <Database className="w-3.5 h-3.5" />
            <span>Sync & Indexing</span>
          </button>
        </nav>

        {/* Right: Actions / Theme Toggle / Metadata */}
        <div className="flex items-center space-x-2.5 z-10">
          <button
            onClick={toggleTheme}
            title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
            className="p-1.5 rounded-md text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100 bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 transition-colors"
          >
            {theme === 'dark' ? <Sun className="w-3.5 h-3.5 text-amber-400" /> : <Moon className="w-3.5 h-3.5 text-sky-600" />}
          </button>

          <a
            href="/api/v1/docs"
            target="_blank"
            rel="noreferrer"
            className="text-[11px] font-mono text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200 px-2 py-1 rounded bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 transition-colors"
          >
            API Docs ↗
          </a>
        </div>
      </header>

      {/* Main Content Area - Preserves Tab State and Active Streams */}
      <main className="flex-1 flex flex-col overflow-hidden max-w-7xl w-full mx-auto p-4">
        <div className={`flex-1 h-full min-h-0 ${activeTab === 'qa' ? 'flex flex-col' : 'hidden'}`}>
          <QAAssistant
            selectedRepo={selectedRepo}
            onSelectPR={handleSelectPRFromQA}
          />
        </div>

        <div className={`flex-1 h-full min-h-0 ${activeTab === 'prs' ? 'flex flex-col' : 'hidden'}`}>
          <PRInspector
            selectedRepo={selectedRepo}
            targetPrNumber={selectedPrNumber}
            onRefreshRepos={fetchRepositories}
          />
        </div>

        <div className={`flex-1 h-full min-h-0 ${activeTab === 'repositories' ? 'flex flex-col' : 'hidden'}`}>
          <RepoManager
            repositories={repositories}
            onRefresh={fetchRepositories}
            selectedRepo={selectedRepo}
            onSelectRepo={setSelectedRepo}
          />
        </div>
      </main>
    </div>
  );
}

