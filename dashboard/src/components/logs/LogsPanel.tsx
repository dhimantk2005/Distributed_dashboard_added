import { useEffect, useMemo, useRef, useState } from 'react';
import type { LogEntry, LogLevel, TerminalLine } from '../../types';
import {
  Terminal,
  AlertTriangle,
  CheckCircle2,
  Bug,
  Filter,
  Search,
} from 'lucide-react';

interface LogsPanelProps {
  logs: LogEntry[];
  terminalLines: TerminalLine[];
}

type FilterLevel = LogLevel | 'all';

const levelMeta: Record<LogLevel, { label: string; color: string }> = {
  info:    { label: 'Info',    color: 'text-sky-400' },
  warning: { label: 'Warning', color: 'text-amber-400' },
  error:   { label: 'Error',   color: 'text-red-400' },
  debug:   { label: 'Debug',   color: 'text-surface-400' },
  success: { label: 'Success', color: 'text-emerald-400' },
};

export default function LogsPanel({ logs, terminalLines }: LogsPanelProps) {
  const [filter, setFilter] = useState<FilterLevel>('all');
  const [query, setQuery] = useState('');
  const [autoScroll, setAutoScroll] = useState(true);
  const [view, setView] = useState<'parsed' | 'raw'>('parsed');
  const scrollerRef = useRef<HTMLDivElement | null>(null);

  const filteredLogs = useMemo(() => {
    const q = query.trim().toLowerCase();
    return logs.filter((entry) => {
      if (filter !== 'all' && entry.level !== filter) return false;
      if (!q) return true;
      const haystack = `${entry.message} ${entry.rank}`.toLowerCase();
      return haystack.includes(q);
    });
  }, [logs, filter, query]);

  const stats = useMemo(() => {
    const total = logs.length;
    const errors = logs.filter((l) => l.level === 'error').length;
    const warnings = logs.filter((l) => l.level === 'warning').length;
    const last = logs[logs.length - 1]?.timestamp ?? null;
    return { total, errors, warnings, last };
  }, [logs]);

  const rawStats = useMemo(() => {
    const total = terminalLines.length;
    const last = terminalLines[terminalLines.length - 1]?.timestamp ?? null;
    return { total, last };
  }, [terminalLines]);

  useEffect(() => {
    if (!autoScroll) return;
    const scroller = scrollerRef.current;
    if (scroller) {
      scroller.scrollTop = scroller.scrollHeight;
    }
  }, [filteredLogs, autoScroll]);

  const formatTime = (ts: number) =>
    new Date(ts).toLocaleTimeString('en-US', { hour12: false });

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Logs</h1>
          <p className="text-sm text-surface-400 mt-1">
            Streaming backend events and training output
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-xs text-surface-500">
            {stats.last
              ? `Last update: ${formatTime(stats.last)}`
              : 'No log data yet'}
          </div>
          <button
            type="button"
            onClick={() => setAutoScroll((v) => !v)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all border ${
              autoScroll
                ? 'bg-emerald-400/10 text-emerald-300 border-emerald-400/20'
                : 'bg-surface-800/40 text-surface-400 border-surface-700/60'
            }`}
          >
            Auto-scroll: {autoScroll ? 'On' : 'Off'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatCard icon={<Terminal size={18} />} label="Total" value={stats.total} color="text-accent-400" />
        <StatCard icon={<AlertTriangle size={18} />} label="Warnings" value={stats.warnings} color="text-amber-400" />
        <StatCard icon={<Bug size={18} />} label="Errors" value={stats.errors} color="text-red-400" />
        <StatCard icon={<CheckCircle2 size={18} />} label="Status" value={stats.errors > 0 ? 'Attention' : 'Healthy'} color={stats.errors > 0 ? 'text-red-400' : 'text-emerald-400'} />
      </div>

      <div className="glass-card p-4 flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2 text-xs text-surface-400">
          <Filter size={14} />
          View
        </div>
        <div className="flex gap-2">
          <FilterPill label="Parsed" active={view === 'parsed'} onClick={() => setView('parsed')} />
          <FilterPill label="Raw Terminal" active={view === 'raw'} onClick={() => setView('raw')} />
        </div>

        {view === 'parsed' && (
          <>
            <div className="flex items-center gap-2 text-xs text-surface-400 ml-4">
              <Filter size={14} />
              Filter
            </div>
            <div className="flex flex-wrap gap-2">
              <FilterPill label="All" active={filter === 'all'} onClick={() => setFilter('all')} />
              {Object.keys(levelMeta).map((level) => {
                const entryLevel = level as LogLevel;
                return (
                  <FilterPill
                    key={entryLevel}
                    label={levelMeta[entryLevel].label}
                    active={filter === entryLevel}
                    onClick={() => setFilter(entryLevel)}
                  />
                );
              })}
            </div>
            <div className="ml-auto flex items-center gap-2 w-full md:w-auto">
              <div className="relative w-full md:w-64">
                <Search size={14} className="absolute left-3 top-2.5 text-surface-500" />
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search logs..."
                  className="w-full bg-surface-900/50 border border-surface-700/60 rounded-lg pl-9 pr-3 py-2 text-xs text-surface-200 placeholder:text-surface-600 focus:outline-none focus:border-accent-500/50"
                />
              </div>
            </div>
          </>
        )}

        {view === 'raw' && (
          <div className="ml-auto text-xs text-surface-500">
            {rawStats.last
              ? `Last update: ${formatTime(rawStats.last)}`
              : 'No terminal output yet'}
          </div>
        )}
      </div>

      <div className="glass-card p-4">
        <div
          ref={scrollerRef}
          className="log-terminal overflow-y-auto rounded-lg bg-black/40"
          style={{ maxHeight: '520px' }}
        >
          {view === 'parsed' && (
            filteredLogs.length === 0 ? (
              <div className="p-6 text-sm text-surface-500 flex items-center gap-2">
                <Terminal size={16} />
                No log entries match your filters.
              </div>
            ) : (
              filteredLogs.map((entry) => (
                <div
                  key={entry.id}
                  className={`log-entry log-${entry.level} flex gap-3 text-surface-200`}
                >
                  <span className="text-surface-500 w-[88px] shrink-0">
                    {formatTime(entry.timestamp)}
                  </span>
                  <span className={`w-[72px] shrink-0 ${levelMeta[entry.level].color}`}>
                    {levelMeta[entry.level].label}
                  </span>
                  <span className="text-surface-400 w-[64px] shrink-0">
                    Rank {entry.rank}
                  </span>
                  <span className="break-words">{entry.message}</span>
                </div>
              ))
            )
          )}

          {view === 'raw' && (
            terminalLines.length === 0 ? (
              <div className="p-6 text-sm text-surface-500 flex items-center gap-2">
                <Terminal size={16} />
                No terminal output yet.
              </div>
            ) : (
              terminalLines.map((entry) => (
                <div key={entry.id} className="log-entry flex gap-3 text-surface-200">
                  <span className="text-surface-500 w-[88px] shrink-0">
                    {formatTime(entry.timestamp)}
                  </span>
                  <span className="text-surface-400 w-[64px] shrink-0">
                    Rank {entry.rank}
                  </span>
                  <span className="break-words">{entry.message}</span>
                </div>
              ))
            )
          )}
        </div>
      </div>
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  value: number | string;
  color: string;
}) {
  return (
    <div className="glass-card p-4">
      <div className={`flex items-center gap-2 mb-2 ${color}`}>
        {icon}
        <span className="text-xs font-medium text-surface-400">{label}</span>
      </div>
      <div className="text-lg font-bold text-white">{value}</div>
    </div>
  );
}

function FilterPill({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all border ${
        active
          ? 'bg-accent-500/20 text-accent-300 border-accent-500/30'
          : 'bg-surface-800/40 text-surface-400 border-surface-700/60 hover:text-surface-200'
      }`}
    >
      {label}
    </button>
  );
}
