import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  Area,
  AreaChart,
  Legend,
} from 'recharts';
import type { TrainingState } from '../../types';
import {
  TrendingDown,
  TrendingUp,
  Clock,
  Layers,
  Gauge,
  BarChart3,
  Cpu,
} from 'lucide-react';

interface TrainingDashboardProps {
  training: TrainingState;
}

const RANK_COLORS = ['#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444'];
const GLOBAL_COLOR = '#34d399';

export default function TrainingDashboard({ training }: TrainingDashboardProps) {

  const lastEpoch = training.epochHistory.length > 0
    ? training.epochHistory[training.epochHistory.length - 1]
    : null;

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}m ${s}s`;
  };

  // ── Chart data from real backend data ─────────────────────────────

  // Loss and accuracy charts: global line from epoch history
  const lossChartData = training.epochHistory.map((eh) => ({
    epoch:  eh.epoch,
    global: parseFloat(eh.loss.toFixed(4)),
  }));

  const accChartData = training.epochHistory.map((eh) => ({
    epoch:  eh.epoch,
    global: parseFloat(eh.accuracy.toFixed(1)),
  }));

  // Per-node throughput from current perNodeMetrics
  const throughputData = training.perNodeMetrics.map((m) => ({
    name:       `Rank ${m.rank}`,
    throughput: m.throughput,
    rank:       m.rank,
  }));

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Training Dashboard</h1>
        <p className="text-sm text-surface-400 mt-1">
          Live metrics from the real training process
        </p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <KpiCard
          icon={<Layers size={18} />}
          label="Current Epoch"
          value={
            training.totalEpochs > 0
              ? `${training.currentEpoch} / ${training.totalEpochs}`
              : '—'
          }
          color="text-accent-400"
        />
        <KpiCard
          icon={<TrendingDown size={18} />}
          label="Global Loss"
          value={training.globalLoss > 0 ? training.globalLoss.toFixed(4) : '—'}
          color="text-red-400"
        />
        <KpiCard
          icon={<TrendingUp size={18} />}
          label="Global Accuracy"
          value={training.globalAccuracy > 0 ? `${training.globalAccuracy.toFixed(1)}%` : '—'}
          color="text-emerald-400"
        />
        <KpiCard
          icon={<Clock size={18} />}
          label="Elapsed Time"
          value={formatTime(training.elapsedTime)}
          color="text-amber-400"
        />
      </div>

      {/* Performance KPIs */}
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
        <KpiCard
          icon={<Gauge size={18} />}
          label="Global Throughput"
          value={lastEpoch?.throughput ? `${lastEpoch.throughput.toFixed(0)} s/s` : '—'}
          color="text-accent-400"
        />
        <KpiCard
          icon={<Clock size={18} />}
          label="Avg Batch Time"
          value={lastEpoch?.avgBatchTime ? `${lastEpoch.avgBatchTime.toFixed(4)}s` : '—'}
          color="text-surface-300"
        />
        <KpiCard
          icon={<Cpu size={18} />}
          label="Max GPU Mem"
          value={lastEpoch?.maxGpuMemMb ? `${lastEpoch.maxGpuMemMb.toFixed(0)} MB` : '—'}
          color="text-emerald-400"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">

        {/* Loss Chart */}
        <div className="glass-card p-5">
          <div className="flex items-center gap-2 mb-4">
            <TrendingDown size={16} className="text-red-400" />
            <span className="text-sm font-semibold text-white">Loss vs Epoch</span>
            <span className="ml-auto text-xs text-surface-500">real data</span>
          </div>
          <div style={{ height: 280 }}>
            {lossChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={lossChartData}>
                  <defs>
                    <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={GLOBAL_COLOR} stopOpacity={0.3} />
                      <stop offset="100%" stopColor={GLOBAL_COLOR} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="epoch" tick={{ fill: '#6b7280', fontSize: 11 }} axisLine={{ stroke: 'rgba(255,255,255,0.08)' }} />
                  <YAxis tick={{ fill: '#6b7280', fontSize: 11 }} axisLine={{ stroke: 'rgba(255,255,255,0.08)' }} />
                  <Tooltip
                    contentStyle={{ background: 'rgba(15,17,23,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 12 }}
                    labelStyle={{ color: '#fff' }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11, color: '#9ca3af' }} />
                  <Area
                    type="monotone"
                    dataKey="global"
                    stroke={GLOBAL_COLOR}
                    strokeWidth={2.5}
                    fill="url(#lossGrad)"
                    name="Global Loss"
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <EmptyChart label="Loss data will appear as epochs complete" />
            )}
          </div>
        </div>

        {/* Accuracy Chart */}
        <div className="glass-card p-5">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp size={16} className="text-emerald-400" />
            <span className="text-sm font-semibold text-white">Accuracy vs Epoch</span>
            <span className="ml-auto text-xs text-surface-500">real data</span>
          </div>
          <div style={{ height: 280 }}>
            {accChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={accChartData}>
                  <defs>
                    <linearGradient id="accGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={GLOBAL_COLOR} stopOpacity={0.3} />
                      <stop offset="100%" stopColor={GLOBAL_COLOR} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="epoch" tick={{ fill: '#6b7280', fontSize: 11 }} axisLine={{ stroke: 'rgba(255,255,255,0.08)' }} />
                  <YAxis domain={[0, 100]} tick={{ fill: '#6b7280', fontSize: 11 }} axisLine={{ stroke: 'rgba(255,255,255,0.08)' }} />
                  <Tooltip
                    contentStyle={{ background: 'rgba(15,17,23,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 12 }}
                    labelStyle={{ color: '#fff' }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11, color: '#9ca3af' }} />
                  <Area
                    type="monotone"
                    dataKey="global"
                    stroke={GLOBAL_COLOR}
                    strokeWidth={2.5}
                    fill="url(#accGrad)"
                    name="Global Accuracy %"
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <EmptyChart label="Accuracy data will appear as epochs complete" />
            )}
          </div>
        </div>
      </div>

      {/* Throughput Bar Chart (live — updates each batch) */}
      <div className="glass-card p-5">
        <div className="flex items-center gap-2 mb-4">
          <Gauge size={16} className="text-accent-400" />
          <span className="text-sm font-semibold text-white">Per-Node Throughput</span>
          <span className="text-xs text-surface-500 ml-auto">samples/sec · live</span>
        </div>
        <div style={{ height: 200 }}>
          {throughputData.some((d) => d.throughput > 0) ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={throughputData} barSize={48}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="name" tick={{ fill: '#6b7280', fontSize: 11 }} axisLine={{ stroke: 'rgba(255,255,255,0.08)' }} />
                <YAxis tick={{ fill: '#6b7280', fontSize: 11 }} axisLine={{ stroke: 'rgba(255,255,255,0.08)' }} />
                <Tooltip
                  contentStyle={{ background: 'rgba(15,17,23,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 12 }}
                  labelStyle={{ color: '#fff' }}
                />
                <Bar dataKey="throughput" name="Throughput" radius={[4, 4, 0, 0]}>
                  {throughputData.map((entry, index) => (
                    <Cell key={entry.rank} fill={RANK_COLORS[index % RANK_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <EmptyChart label="Throughput will appear once training starts" />
          )}
        </div>
      </div>

      {/* Per-Node Metrics Table */}
      <div className="glass-card p-5">
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 size={16} className="text-accent-400" />
          <span className="text-sm font-semibold text-white">Per-Node Metrics</span>
          <span className="text-xs text-surface-500 ml-auto">updated per batch</span>
        </div>
        {training.perNodeMetrics.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-surface-400 border-b border-surface-800">
                  <th className="pb-3 pr-4 font-medium">Rank</th>
                  <th className="pb-3 pr-4 font-medium">Loss</th>
                  <th className="pb-3 pr-4 font-medium">Accuracy</th>
                  <th className="pb-3 pr-4 font-medium">Throughput</th>
                  <th className="pb-3 font-medium">Batches Done</th>
                </tr>
              </thead>
              <tbody className="text-surface-200">
                {training.perNodeMetrics.map((m) => (
                  <tr
                    key={m.rank}
                    className="border-b border-surface-800/50 hover:bg-white/[0.02] transition-colors"
                  >
                    <td className="py-2.5 pr-4">
                      <div className="flex items-center gap-2">
                        <div
                          className="w-2 h-2 rounded-full"
                          style={{ background: RANK_COLORS[m.rank % RANK_COLORS.length] }}
                        />
                        <span className="font-medium">Rank {m.rank}</span>
                        {m.rank === 0 && (
                          <span className="text-[10px] text-amber-400 font-semibold">MASTER</span>
                        )}
                      </div>
                    </td>
                    <td className="py-2.5 pr-4 font-mono text-xs">
                      {m.loss > 0 ? m.loss.toFixed(4) : '—'}
                    </td>
                    <td className="py-2.5 pr-4 font-mono text-xs">
                      {m.accuracy > 0 ? `${m.accuracy.toFixed(1)}%` : '—'}
                    </td>
                    <td className="py-2.5 pr-4 font-mono text-xs">
                      {m.throughput > 0 ? `${m.throughput.toFixed(0)} s/s` : '—'}
                    </td>
                    <td className="py-2.5 font-mono text-xs">{m.batchesCompleted}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center text-surface-500 text-sm py-8">
            Node metrics will appear once training is running
          </div>
        )}
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────

function KpiCard({
  icon, label, value, color,
}: {
  icon:  React.ReactNode;
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="glass-card p-4">
      <div className={`flex items-center gap-2 mb-2 ${color}`}>
        {icon}
        <span className="text-xs font-medium text-surface-400">{label}</span>
      </div>
      <div className="text-xl font-bold text-white">{value}</div>
    </div>
  );
}

function EmptyChart({ label }: { label: string }) {
  return (
    <div className="h-full flex items-center justify-center text-surface-500 text-sm">
      {label}
    </div>
  );
}
