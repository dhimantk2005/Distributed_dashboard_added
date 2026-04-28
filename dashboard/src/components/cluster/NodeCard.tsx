import type { ClusterNode } from '../../types';
import { Crown, Server, Cpu, Thermometer, Zap, RefreshCw, XCircle } from 'lucide-react';

interface NodeCardProps {
  node: ClusterNode;
}

const statusConfig: Record<string, { color: string; bgColor: string; label: string; dotClass: string }> = {
  connected: {
    color: 'text-emerald-400',
    bgColor: 'bg-emerald-400/10',
    label: 'Connected',
    dotClass: 'bg-emerald-400',
  },
  training: {
    color: 'text-emerald-400',
    bgColor: 'bg-emerald-400/10',
    label: 'Training',
    dotClass: 'bg-emerald-400',
  },
  retrying: {
    color: 'text-amber-400',
    bgColor: 'bg-amber-400/10',
    label: 'Retrying',
    dotClass: 'bg-amber-400',
  },
  failed: {
    color: 'text-red-400',
    bgColor: 'bg-red-400/10',
    label: 'Failed',
    dotClass: 'bg-red-400',
  },
  idle: {
    color: 'text-surface-400',
    bgColor: 'bg-surface-400/10',
    label: 'Idle',
    dotClass: 'bg-surface-500',
  },
};

export default function NodeCard({ node }: NodeCardProps) {
  const st = statusConfig[node.status] || statusConfig.idle;
  const batchProgress = node.totalBatches > 0 ? (node.currentBatch / node.totalBatches) * 100 : 0;
  const epochProgress = node.totalEpochs > 0 ? ((node.currentEpoch + batchProgress / 100) / node.totalEpochs) * 100 : 0;

  const cardGlow =
    node.status === 'training'
      ? 'animate-pulse-glow'
      : node.status === 'failed'
      ? 'animate-fail-glow'
      : '';

  return (
    <div
      className={`glass-card glass-card-hover p-5 animate-slide-in ${cardGlow}`}
      style={{ animationDelay: `${node.rank * 80}ms` }}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div
            className={`w-10 h-10 rounded-lg flex items-center justify-center ${
              node.isMaster
                ? 'bg-gradient-to-br from-amber-500/20 to-orange-500/20 border border-amber-500/30'
                : 'bg-surface-800/80 border border-surface-700/50'
            }`}
          >
            {node.isMaster ? (
              <Crown size={18} className="text-amber-400" />
            ) : (
              <Server size={18} className="text-surface-300" />
            )}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-white">
                Rank {node.rank}
              </span>
              {node.isMaster && (
                <span className="text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-400">
                  Master
                </span>
              )}
            </div>
            <div className="text-xs text-surface-400 mt-0.5">{node.hostname}</div>
          </div>
        </div>

        {/* Status Pill */}
        <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full ${st.bgColor}`}>
          <div className="status-dot-wrapper">
            <div className={`w-1.5 h-1.5 rounded-full ${st.dotClass}`} />
            {(node.status === 'training' || node.status === 'retrying') && (
              <div className={`status-dot-ping ${st.dotClass} opacity-40`} />
            )}
          </div>
          <span className={`text-[11px] font-medium ${st.color}`}>
            {st.label}
            {node.status === 'retrying' && node.retryCount > 0 && (
              <span className="ml-1 opacity-70">({node.retryCount})</span>
            )}
          </span>
        </div>
      </div>

      {/* IP Address */}
      <div className="flex items-center gap-2 mb-3 text-xs">
        <span className="text-surface-500">IP</span>
        <code className="text-surface-300 bg-surface-800/60 px-2 py-0.5 rounded font-mono text-[11px]">
          {node.ip}
        </code>
        <span className="text-surface-600 ml-auto">
          {node.backend.toUpperCase()}
        </span>
      </div>

      {/* GPU Info */}
      {node.gpu && (
        <div className="mb-4 p-3 rounded-lg bg-surface-900/50 border border-surface-800/50 space-y-2">
          <div className="flex items-center gap-2 text-xs">
            <Cpu size={12} className="text-accent-400" />
            <span className="text-surface-200 font-medium">{node.gpu.name}</span>
          </div>
          <div className="grid grid-cols-3 gap-2 text-[11px]">
            <div className="flex items-center gap-1 text-surface-400">
              <Zap size={10} />
              <span>{node.gpu.utilization.toFixed(0)}%</span>
            </div>
            <div className="flex items-center gap-1 text-surface-400">
              <Thermometer size={10} />
              <span>{node.gpu.temperature}°C</span>
            </div>
            <div className="text-surface-400 text-right">
              {node.gpu.memoryUsed.toFixed(1)}/{node.gpu.memoryTotal}GB
            </div>
          </div>
          {/* GPU Memory Bar */}
          <div className="h-1.5 bg-surface-800 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-700 ease-out"
              style={{
                width: `${(node.gpu.memoryUsed / node.gpu.memoryTotal) * 100}%`,
                background: `linear-gradient(90deg, var(--color-accent-500), var(--color-accent-400))`,
              }}
            />
          </div>
        </div>
      )}

      {/* Epoch Progress */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-surface-400">Epoch</span>
          <span className="text-surface-200 font-medium">
            {node.currentEpoch + 1} / {node.totalEpochs}
          </span>
        </div>
        <div className="h-1.5 bg-surface-800 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500 ease-out"
            style={{
              width: `${epochProgress}%`,
              background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)',
            }}
          />
        </div>

        {/* Batch Progress */}
        <div className="flex items-center justify-between text-xs">
          <span className="text-surface-400">Batch</span>
          <span className="text-surface-200 font-medium">
            {node.currentBatch} / {node.totalBatches}
          </span>
        </div>
        <div className="h-1 bg-surface-800 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500 ease-out bg-surface-500"
            style={{ width: `${batchProgress}%` }}
          />
        </div>
      </div>

      {/* Throughput */}
      <div className="mt-3 flex items-center justify-between pt-3 border-t border-surface-800/50">
        <div className="flex items-center gap-1.5 text-xs text-surface-400">
          {node.status === 'retrying' ? (
            <RefreshCw size={12} className="text-amber-400 animate-spin" />
          ) : node.status === 'failed' ? (
            <XCircle size={12} className="text-red-400" />
          ) : (
            <Zap size={12} className="text-accent-400" />
          )}
          <span>Throughput</span>
        </div>
        <span className="text-xs font-semibold text-surface-200">
          {node.throughput > 0 ? `${node.throughput} samples/s` : '—'}
        </span>
      </div>
    </div>
  );
}
