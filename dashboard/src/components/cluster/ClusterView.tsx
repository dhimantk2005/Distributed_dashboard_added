import type { ClusterNode } from '../../types';
import NodeCard from './NodeCard';
import {
  Server,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Network,
} from 'lucide-react';

interface ClusterViewProps {
  nodes: ClusterNode[];
}

export default function ClusterView({ nodes }: ClusterViewProps) {
  const totalNodes = nodes.length;
  const healthyNodes = nodes.filter(
    (n) => n.status === 'connected' || n.status === 'training' || n.status === 'idle'
  ).length;
  const warningNodes = nodes.filter((n) => n.status === 'retrying').length;
  const failedNodes = nodes.filter((n) => n.status === 'failed').length;

  const masterNode = nodes.find((n) => n.isMaster);
  const workerNodes = nodes.filter((n) => !n.isMaster);

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">
          Cluster Overview
        </h1>
        <p className="text-sm text-surface-400 mt-1">
          Real-time node status and distributed system health
        </p>
      </div>

      {/* Aggregate Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatCard
          icon={<Server size={18} />}
          label="Total Nodes"
          value={totalNodes.toString()}
          color="text-accent-400"
          bgColor="bg-accent-500/10"
        />
        <StatCard
          icon={<CheckCircle2 size={18} />}
          label="Healthy"
          value={healthyNodes.toString()}
          color="text-emerald-400"
          bgColor="bg-emerald-400/10"
        />
        <StatCard
          icon={<AlertTriangle size={18} />}
          label="Unstable"
          value={warningNodes.toString()}
          color="text-amber-400"
          bgColor="bg-amber-400/10"
        />
        <StatCard
          icon={<XCircle size={18} />}
          label="Failed"
          value={failedNodes.toString()}
          color="text-red-400"
          bgColor="bg-red-400/10"
        />
      </div>

      {/* Topology Visual */}
      <div className="glass-card p-5">
        <div className="flex items-center gap-2 mb-4">
          <Network size={16} className="text-accent-400" />
          <span className="text-sm font-semibold text-white">Cluster Topology</span>
          <span className="text-xs text-surface-500 ml-auto">
            PyTorch DDP • {masterNode?.backend.toUpperCase() || 'NCCL'} Backend
          </span>
        </div>

        {/* SVG Topology */}
        <div className="relative" style={{ minHeight: '120px' }}>
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            style={{ zIndex: 0 }}
          >
            {workerNodes.map((worker, i) => {
              const masterX = 12;
              const masterY = 60;
              const workerX = 35 + i * 22;
              const workerY = 100;

              return (
                <line
                  key={worker.rank}
                  x1={`${masterX}%`}
                  y1={`${masterY}%`}
                  x2={`${workerX}%`}
                  y2={`${workerY}%`}
                  className={`topology-line ${
                    worker.status === 'training' || worker.status === 'connected'
                      ? 'active'
                      : ''
                  }`}
                  style={{
                    stroke:
                      worker.status === 'failed'
                        ? 'var(--color-danger)'
                        : worker.status === 'retrying'
                        ? 'var(--color-warning)'
                        : undefined,
                  }}
                />
              );
            })}
          </svg>

          <div className="relative z-10 flex flex-wrap items-start gap-6">
            {/* Master Badge */}
            <div className="flex flex-col items-center gap-1">
              <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-amber-500/20 to-orange-500/20 border border-amber-500/30 flex items-center justify-center">
                <span className="text-lg font-bold text-amber-400">M</span>
              </div>
              <span className="text-[10px] text-surface-400 font-medium">
                {masterNode?.ip}
              </span>
              <div
                className={`w-2 h-2 rounded-full ${
                  masterNode?.status === 'training'
                    ? 'bg-emerald-400'
                    : 'bg-surface-500'
                }`}
              />
            </div>

            {/* Arrow area */}
            <div className="flex-1 flex items-center justify-center pt-4">
              <span className="text-xs text-surface-500 italic">
                gradient sync via {masterNode?.backend.toUpperCase() || 'NCCL'}
              </span>
            </div>

            {/* Worker Badges */}
            <div className="flex gap-3">
              {workerNodes.map((w) => {
                const borderColor =
                  w.status === 'failed'
                    ? 'border-red-500/40'
                    : w.status === 'retrying'
                    ? 'border-amber-500/40'
                    : 'border-surface-700/50';
                return (
                  <div key={w.rank} className="flex flex-col items-center gap-1">
                    <div
                      className={`w-14 h-14 rounded-xl bg-surface-800/80 border ${borderColor} flex items-center justify-center`}
                    >
                      <span className="text-lg font-bold text-surface-300">
                        W{w.rank}
                      </span>
                    </div>
                    <span className="text-[10px] text-surface-400 font-medium">
                      {w.ip}
                    </span>
                    <div
                      className={`w-2 h-2 rounded-full ${
                        w.status === 'training'
                          ? 'bg-emerald-400'
                          : w.status === 'retrying'
                          ? 'bg-amber-400'
                          : w.status === 'failed'
                          ? 'bg-red-400'
                          : 'bg-surface-500'
                      }`}
                    />
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Node Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        {nodes.map((node) => (
          <NodeCard key={node.rank} node={node} />
        ))}
      </div>
    </div>
  );
}

// ─── Sub-components ─────────────────────────────────────────────────

function StatCard({
  icon,
  label,
  value,
  color,
  bgColor,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  color: string;
  bgColor: string;
}) {
  return (
    <div className="glass-card p-4 flex items-center gap-3">
      <div className={`w-10 h-10 rounded-lg ${bgColor} flex items-center justify-center ${color}`}>
        {icon}
      </div>
      <div>
        <div className="text-xl font-bold text-white">{value}</div>
        <div className="text-xs text-surface-400">{label}</div>
      </div>
    </div>
  );
}
