import React from 'react';
import { Copy, Check } from 'lucide-react';

function CodeBlock({ code, language }) {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative my-3 rounded-lg overflow-hidden border border-slate-800 bg-[#090d16] font-mono text-xs">
      <div className="flex items-center justify-between px-3 py-1.5 bg-slate-900/90 border-b border-slate-800 text-slate-400 text-[11px]">
        <span>{language || 'text'}</span>
        <button
          onClick={handleCopy}
          className="flex items-center space-x-1 hover:text-slate-200 transition-colors"
        >
          {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
          <span>{copied ? 'Copied' : 'Copy'}</span>
        </button>
      </div>
      <pre className="p-3.5 overflow-x-auto text-slate-200 leading-relaxed">
        <code>{code}</code>
      </pre>
    </div>
  );
}

export default function MarkdownRenderer({ content, onSelectPR }) {
  if (!content) return null;

  // Process and format inline markdown elements (bold, code, links, PR tags)
  const formatInline = (text) => {
    // Regex matches: inline code, bold, PR citations like PR #123
    const tokens = [];
    let remaining = text;
    let key = 0;

    while (remaining) {
      // 1. Inline code: `code`
      const codeMatch = remaining.match(/^`([^`]+)`/);
      if (codeMatch) {
        tokens.push(
          <code
            key={key++}
            className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-sky-300 font-mono text-[12px]"
          >
            {codeMatch[1]}
          </code>
        );
        remaining = remaining.slice(codeMatch[0].length);
        continue;
      }

      // 2. Bold text: **bold** or __bold__
      const boldMatch = remaining.match(/^\*\*([^*]+)\*\*/) || remaining.match(/^__([^_]+)__/);
      if (boldMatch) {
        tokens.push(
          <strong key={key++} className="font-semibold text-slate-100">
            {formatInline(boldMatch[1])}
          </strong>
        );
        remaining = remaining.slice(boldMatch[0].length);
        continue;
      }

      // 3. PR references: PR #123 or #123
      const prMatch = remaining.match(/^(PR\s*#(\d+)|#(\d+))/i);
      if (prMatch) {
        const prNum = parseInt(prMatch[2] || prMatch[3], 10);
        tokens.push(
          <button
            key={key++}
            type="button"
            onClick={() => onSelectPR && onSelectPR(prNum)}
            className="inline-flex items-center px-1.5 py-0.2 mx-0.5 rounded font-mono font-bold text-xs bg-sky-950 text-sky-300 border border-sky-800/80 hover:bg-sky-900 transition-colors"
          >
            PR #{prNum}
          </button>
        );
        remaining = remaining.slice(prMatch[0].length);
        continue;
      }

      // Plain character
      const nextSpecial = remaining.search(/[`*#]|PR\s*#/i);
      if (nextSpecial === -1) {
        tokens.push(remaining);
        break;
      } else if (nextSpecial === 0) {
        tokens.push(remaining[0]);
        remaining = remaining.slice(1);
      } else {
        tokens.push(remaining.slice(0, nextSpecial));
        remaining = remaining.slice(nextSpecial);
      }
    }

    return tokens;
  };

  // Split content into blocks (code blocks, lists, headers, paragraphs, tables)
  const lines = content.split('\n');
  const elements = [];
  let inCodeBlock = false;
  let codeBuffer = [];
  let codeLang = '';

  let listBuffer = [];
  let listType = null; // 'ul' | 'ol'
  let tableBuffer = [];

  const flushList = () => {
    if (listBuffer.length > 0) {
      if (listType === 'ul') {
        elements.push(
          <ul key={`ul-${elements.length}`} className="my-3 space-y-2 pl-6 list-disc text-slate-300 text-[14.5px] leading-[1.75] font-sans">
            {listBuffer.map((item, i) => (
              <li key={i} className="pl-1">
                {formatInline(item)}
              </li>
            ))}
          </ul>
        );
      } else {
        elements.push(
          <ol key={`ol-${elements.length}`} className="my-3 space-y-2 pl-6 list-decimal text-slate-300 text-[14.5px] leading-[1.75] font-sans">
            {listBuffer.map((item, i) => (
              <li key={i} className="pl-1">
                {formatInline(item)}
              </li>
            ))}
          </ol>
        );
      }
      listBuffer = [];
      listType = null;
    }
  };

  const parseCells = (rowStr) => {
    const trimmed = rowStr.trim();
    const withoutEdges = trimmed.replace(/^\|/, '').replace(/\|$/, '');
    return withoutEdges.split('|').map((c) => c.trim());
  };

  const flushTable = () => {
    if (tableBuffer.length >= 2) {
      const headerCells = parseCells(tableBuffer[0]);
      const separatorRow = tableBuffer[1];
      const sepCells = parseCells(separatorRow);

      // Check if second row is truly a markdown table separator (contains ---)
      const isValidSeparator = sepCells.every((s) => /^:?-+:?$/.test(s.trim()));

      if (isValidSeparator) {
        const alignments = sepCells.map((s) => {
          const t = s.trim();
          if (t.startsWith(':') && t.endsWith(':')) return 'text-center';
          if (t.endsWith(':')) return 'text-right';
          return 'text-left';
        });

        const dataRows = tableBuffer.slice(2).map((r) => parseCells(r));

        elements.push(
          <div key={`table-${elements.length}`} className="my-4 overflow-x-auto rounded-lg border border-slate-800 bg-[#090d16]/90 shadow-sm">
            <table className="w-full text-left border-collapse text-xs sm:text-[13.5px] font-sans">
              <thead>
                <tr className="border-b border-slate-800 bg-slate-900/90">
                  {headerCells.map((h, hi) => (
                    <th
                      key={hi}
                      className={`px-4 py-2.5 font-mono text-xs font-semibold text-slate-300 tracking-wider ${alignments[hi] || 'text-left'}`}
                    >
                      {formatInline(h)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {dataRows.map((row, ri) => (
                  <tr key={ri} className="hover:bg-slate-900/40 transition-colors">
                    {row.map((cell, ci) => (
                      <td
                        key={ci}
                        className={`px-4 py-3 text-slate-200 leading-relaxed align-top ${alignments[ci] || 'text-left'}`}
                      >
                        {formatInline(cell)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
        tableBuffer = [];
        return;
      }
    }

    // Fallback: If not a valid table structure, render lines as regular text
    if (tableBuffer.length > 0) {
      tableBuffer.forEach((tLine) => {
        elements.push(
          <p key={`p-${elements.length}`} className="my-2.5 text-[14.5px] text-slate-200 leading-[1.75] font-sans">
            {formatInline(tLine)}
          </p>
        );
      });
      tableBuffer = [];
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Check code blocks
    if (line.trim().startsWith('```')) {
      if (inCodeBlock) {
        flushList();
        flushTable();
        elements.push(
          <CodeBlock
            key={`code-${elements.length}`}
            code={codeBuffer.join('\n')}
            language={codeLang}
          />
        );
        inCodeBlock = false;
        codeBuffer = [];
        codeLang = '';
      } else {
        flushList();
        flushTable();
        inCodeBlock = true;
        codeLang = line.trim().replace(/^```/, '').trim();
      }
      continue;
    }

    if (inCodeBlock) {
      codeBuffer.push(line);
      continue;
    }

    // Check table rows
    if (line.trim().startsWith('|') && line.trim().endsWith('|')) {
      flushList();
      tableBuffer.push(line);
      continue;
    } else if (tableBuffer.length > 0) {
      flushTable();
    }

    // Check horizontal rules
    if (line.trim() === '---' || line.trim() === '***') {
      flushList();
      flushTable();
      elements.push(<hr key={`hr-${elements.length}`} className="my-5 border-slate-800" />);
      continue;
    }

    // Check headings
    if (line.startsWith('# ')) {
      flushList();
      flushTable();
      elements.push(
        <h1 key={`h1-${elements.length}`} className="text-xl font-bold text-slate-100 tracking-tight mt-5 mb-2.5 font-sans">
          {formatInline(line.slice(2))}
        </h1>
      );
      continue;
    }
    if (line.startsWith('## ')) {
      flushList();
      flushTable();
      elements.push(
        <h2 key={`h2-${elements.length}`} className="text-base font-bold text-slate-100 tracking-tight mt-4 mb-2 border-b border-slate-800/80 pb-1.5 font-sans">
          {formatInline(line.slice(3))}
        </h2>
      );
      continue;
    }
    if (line.startsWith('### ')) {
      flushList();
      flushTable();
      elements.push(
        <h3 key={`h3-${elements.length}`} className="text-sm font-semibold text-sky-400 tracking-tight mt-3.5 mb-1.5 font-sans">
          {formatInline(line.slice(4))}
        </h3>
      );
      continue;
    }

    // Check blockquote
    if (line.startsWith('> ')) {
      flushList();
      flushTable();
      elements.push(
        <blockquote
          key={`bq-${elements.length}`}
          className="my-3 border-l-2 border-sky-500/80 pl-3.5 py-1.5 bg-slate-900/40 text-slate-300 italic text-[14px] leading-relaxed font-sans rounded-r"
        >
          {formatInline(line.slice(2))}
        </blockquote>
      );
      continue;
    }

    // Check unordered list item
    const ulMatch = line.match(/^(\s*)[-*+]\s+(.*)$/);
    if (ulMatch) {
      if (listType !== 'ul') {
        flushList();
        flushTable();
        listType = 'ul';
      }
      listBuffer.push(ulMatch[2]);
      continue;
    }

    // Check ordered list item
    const olMatch = line.match(/^(\s*)\d+\.\s+(.*)$/);
    if (olMatch) {
      if (listType !== 'ol') {
        flushList();
        flushTable();
        listType = 'ol';
      }
      listBuffer.push(olMatch[2]);
      continue;
    }

    // Regular paragraph or empty line
    if (!line.trim()) {
      flushList();
      flushTable();
      continue;
    }

    flushList();
    flushTable();
    elements.push(
      <p key={`p-${elements.length}`} className="my-2.5 text-[14.5px] text-slate-200 leading-[1.75] font-sans">
        {formatInline(line)}
      </p>
    );
  }

  // Flush any remaining elements
  if (inCodeBlock && codeBuffer.length > 0) {
    elements.push(
      <CodeBlock
        key={`code-${elements.length}`}
        code={codeBuffer.join('\n')}
        language={codeLang}
      />
    );
  }
  flushList();
  flushTable();

  return <div className="markdown-body space-y-1">{elements}</div>;
}

