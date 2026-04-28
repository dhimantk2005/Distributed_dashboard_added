import { useState } from 'react';
import type { DiagnosticsResult } from '../../types';
import {
  Cpu,
  RefreshCw,
  Terminal,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Zap,
  Monitor,
  Loader2,
} from 'lucide-react';

interface DiagnosticsPanelProps {
  onRun:     () => Promise<DiagnosticsResult>;
  result:    DiagnosticsResult | null;
  isRunning: boolean;
}

export default function DiagnosticsPanel({ onRun, result, isRunning }: DiagnosticsPanelProps) {
  const [error, setError] = useState<string | null>(null);

  const handleRun = async () => {
    setError(null);
    try {
      await onRun();
    } catch (e) {
      setError(`Failed to run diagnostics: ${e}`);
    }
  };

  const ts = result?.timestamp
    ? new Date(result.timestamp).toLocaleTimeString('en-US', { hour12: false })
    : null;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Diagnostics</h1>
          <p className="text-sm text-surface-400 mt-1">
            GPU environment and system health — powered by gpu_test.py
          </p>
        </div>
        <div className="flex items-center gap-3">
          {ts && (
            <span className="text-xs text-surface-500">Last run: {ts}</span>
          )}
          <button
            id="run-diagnostics-btn"
            onClick={handleRun}
            disabled={isRunning}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold transition-all ${
              isRunning
                ? 'bg-surface-800 text-surface-500 cursor-not-allowed'
                : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white shadow-lg shadow-blue-500/20 cursor-pointer'
            }`}
          >
            {isRunning
              ? <Loader2 size={16} className="animate-spin" />
              : <RefreshCw size={16} />
            }
            {isRunning ? 'Running…' : 'Run Diagnostics'}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="glass-card p-4 border-l-4 border-red-500 flex items-center gap-3">
          <XCircle size={18} className="text-red-400 shrink-0" />
          <span className="text-sm text-red-300">{error}</span>
        </div>
      )}

      {/* Backend error in result */}
      {result?.error && (
        <div className="glass-card p-4 border-l-4 border-amber-500 flex items-center gap-3">
          <AlertTriangle size={18} className="text-amber-400 shrink-0" />
          <div>
            <div className="text-sm font-medium text-white">Diagnostics reported an error</div>
            <div className="text-xs text-surface-400 mt-0.5">{result.error}</div>
          </div>
        </div>
      )}

      {/* Running placeholder */}
      {isRunning && !result && (
        <div className="glass-card p-10 flex flex-col items-center gap-4 text-surface-400">
          <Loader2 size={40} className="animate-spin text-accent-400" />
          <span className="text-sm">Running gpu_test.py — this may take a few seconds…</span>
        </div>
      )}

      {/* Empty state */}
      {!isRunning && !result && (
        <div className="glass-card p-16 flex flex-col items-center gap-4 text-surface-500">
          <Cpu size={48} className="opacity-20" />
          <div className="text-center">
            <p className="text-sm font-medium text-surface-400">No diagnostics data yet</p>
            <p className="text-xs mt-1">
              Click <span className="text-accent-400">Run Diagnostics</span> to inspect GPU and system environment
            </p>
          </div>
        </div>
      )}

      {/* Results */}
      {result && !isRunning && (
        <>
          {/* Status cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <DiagCard
              icon={<Zap size={18} />}
              label="CUDA Available"
              value={result.torch_cuda ? 'Yes' : 'No'}
              status={result.torch_cuda ? 'ok' : 'fail'}
            />
            <DiagCard
              icon={<Cpu size={18} />}
              label="GPU Count"
              value={result.gpu_count > 0 ? String(result.gpu_count) : 'None'}
              status={result.gpu_count > 0 ? 'ok' : 'warn'}
            />
            <DiagCard
              icon={<CheckCircle2 size={18} />}
              label="cuDNN"
              value={result.cudnn_enabled ? result.cudnn_version : 'Not Available'}
              status={result.cudnn_enabled ? 'ok' : 'warn'}
            />
            <DiagCard
              icon={<Monitor size={18} />}
              label="Environment"
              value={result.is_wsl ? 'WSL / Windows' : 'Native Linux'}
              status="info"
              note={result.is_wsl ? 'NCCL may be limited' : undefined}
            />
          </div>

          {/* Version row */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <InfoRow label="PyTorch Version"  value={result.torch_version || 'N/A'} />
            <InfoRow label="CUDA Version"     value={result.cuda_version  || 'N/A'} />
          </div>

          {/* GPU list */}
          {result.gpu_names.length > 0 && (
            <div className="glass-card p-5">
              <div className="flex items-center gap-2 mb-4">
                <Cpu size={16} className="text-accent-400" />
                <span className="text-sm font-semibold text-white">
                  Detected GPUs ({result.gpu_count})
                </span>
              </div>
              <div className="space-y-2">
                {result.gpu_names.map((name, i) => (
                  <div
                    key={i}
                    className="flex items-center gap-3 p-3 rounded-lg bg-surface-900/50 border border-surface-800/50"
                  >
                    <div className="w-8 h-8 rounded-lg bg-emerald-400/10 flex items-center justify-center">
                      <span className="text-xs font-bold text-emerald-400">{i}</span>
                    </div>
                    <span className="text-sm text-surface-200 font-medium">{name}</span>
                    <span className="ml-auto text-xs text-emerald-400 font-semibold">AVAILABLE</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* nvidia-smi status */}
          <div className="glass-card p-4 flex items-center gap-3">
            <div
              className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                result.nvidia_smi_available ? 'bg-emerald-400/10' : 'bg-red-400/10'
              }`}
            >
              {result.nvidia_smi_available
                ? <CheckCircle2 size={16} className="text-emerald-400" />
                : <XCircle size={16} className="text-red-400" />
              }
            </div>
            <span className="text-sm text-surface-200">
              nvidia-smi: {result.nvidia_smi_available ? 'Available' : 'Not found'}
            </span>
          </div>

          {/* Raw terminal output */}
          <div className="glass-card p-5">
            <div className="flex items-center gap-2 mb-4">
              <Terminal size={16} className="text-surface-400" />
              <span className="text-xs font-medium text-surface-400">Raw Output — gpu_test.py</span>
            </div>
            <div
              className="log-terminal overflow-y-auto rounded-lg bg-black/40 p-4"
              style={{ maxHeight: '400px' }}
            >
              {result.raw_output
                ? result.raw_output.split('\n').map((line, i) => (
                    <div key={i} className="text-surface-300 whitespace-pre-wrap break-all leading-relaxed">
                      {line || '\u00A0'}
                    </div>
                  ))
                : <span className="text-surface-600">No output captured</span>
              }
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────

type DiagStatus = 'ok' | 'warn' | 'fail' | 'info';

const statusStyles: Record<DiagStatus, { icon: string; bg: string; border: string }> = {
  ok:   { icon: 'text-emerald-400', bg: 'bg-emerald-400/10', border: 'border-emerald-400/20' },
  warn: { icon: 'text-amber-400',   bg: 'bg-amber-400/10',   border: 'border-amber-400/20'   },
  fail: { icon: 'text-red-400',     bg: 'bg-red-400/10',     border: 'border-red-400/20'     },
  info: { icon: 'text-accent-400',  bg: 'bg-accent-500/10',  border: 'border-accent-500/20'  },
};

function DiagCard({
  icon, label, value, status, note,
}: {
  icon:    React.ReactNode;
  label:   string;
  value:   string;
  status:  DiagStatus;
  note?:   string;
}) {
  const s = statusStyles[status];
  return (
    <div className={`glass-card p-4 border ${s.border}`}>
      <div className={`flex items-center gap-2 mb-2 ${s.icon}`}>
        {icon}
        <span className="text-xs font-medium text-surface-400">{label}</span>
      </div>
      <div className="text-lg font-bold text-white">{value}</div>
      {note && <div className="text-[10px] text-surface-500 mt-1">{note}</div>}
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="glass-card px-4 py-3 flex items-center justify-between">
      <span className="text-xs text-surface-400">{label}</span>
      <code className="text-xs font-mono text-surface-200 bg-surface-800/60 px-2 py-0.5 rounded">
        {value}
      </code>
    </div>
  );
}
