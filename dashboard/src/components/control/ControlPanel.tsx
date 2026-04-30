import { useState } from 'react';
import type { TrainingConfig } from '../../types';
import type { StartTrainingResult } from '../../hooks/useBackend';
import {
  Play,
  UserPlus,
  Square,
  AlertCircle,
  CheckCircle2,
  Settings,
  Network,
  Cpu,
  Loader2,
} from 'lucide-react';

interface ControlPanelProps {
  isRunning: boolean;
  onStart:   (config: TrainingConfig) => Promise<StartTrainingResult>;
  onStop:    () => Promise<void>;
}

const defaultConfig: TrainingConfig = {
  role:        'master',
  rank:        0,
  worldSize:   1,
  masterAddr:  '127.0.0.1',
  masterPort:  '29500',
  backend:     'gloo',
  epochs:      5,
  batchSize:   64,
  learningRate: 0.001,
  initMethod:  'env://',
  dataDir:     './data',
  saveDir:     './checkpoints',
};

export default function ControlPanel({ isRunning, onStart, onStop }: ControlPanelProps) {
  const [config, setConfig]       = useState<TrainingConfig>(defaultConfig);
  const [errors, setErrors]       = useState<Record<string, string>>({});
  const [submitted, setSubmitted] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [startError, setStartError]     = useState<string | null>(null);
  const [isStopping, setIsStopping]     = useState(false);

  // ── Validation ────────────────────────────────────────────────────

  const validate = (): boolean => {
    const errs: Record<string, string> = {};
    if (config.worldSize < 1) errs.worldSize = 'World size must be ≥ 1';
    if (config.rank < 0 || config.rank >= config.worldSize)
      errs.rank = `Rank must be in [0, ${config.worldSize - 1}]`;
    if (!config.masterAddr.trim()) errs.masterAddr = 'Master address is required';
    const port = parseInt(config.masterPort);
    if (isNaN(port) || port < 1024 || port > 65535)
      errs.masterPort = 'Port must be between 1024 and 65535';
    if (config.epochs < 1)      errs.epochs       = 'At least 1 epoch required';
    if (config.batchSize < 1)   errs.batchSize    = 'Batch size must be positive';
    if (config.learningRate <= 0) errs.learningRate = 'Learning rate must be positive';
    setErrors(errs);
    return Object.keys(errs).length === 0;
  };

  // ── Handlers ──────────────────────────────────────────────────────

  const handleStart = async () => {
    if (!validate()) return;
    setIsSubmitting(true);
    setStartError(null);
    try {
      const result = await onStart(config);
      if (result.success) {
        setSubmitted(true);
      } else {
        setStartError(result.error ?? 'Failed to start training');
      }
    } catch (e) {
      setStartError(String(e));
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleStop = async () => {
    setIsStopping(true);
    try { await onStop(); }
    finally { setIsStopping(false); }
  };

  const update = <K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) => {
    setConfig((prev) => {
      const next = { ...prev, [key]: value };
      if (key === 'role') next.rank = value === 'master' ? 0 : 1;
      return next;
    });
    if (errors[key]) setErrors((prev) => { const n = { ...prev }; delete n[key]; return n; });
  };

  // ── Generated command ─────────────────────────────────────────────

  const cmd = [
    `python train.py`,
    `--rank ${config.rank}`,
    `--world-size ${config.worldSize}`,
    `--master-addr ${config.masterAddr}`,
    `--master-port ${config.masterPort}`,
    `--backend ${config.backend}`,
    `--epochs ${config.epochs}`,
    `--batch-size ${config.batchSize}`,
    `--lr ${config.learningRate}`,
    `--init-method ${config.initMethod}`,
  ].join(' ');

  // ── Render ────────────────────────────────────────────────────────

  return (
    <div className="space-y-6 animate-fade-in max-w-3xl">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Control Panel</h1>
        <p className="text-sm text-surface-400 mt-1">
          Configure and launch distributed training — runs the real train.py
        </p>
      </div>

      {/* Status Banners */}
      {isRunning && (
        <div className="glass-card p-4 border-l-4 border-emerald-400 flex items-center gap-3">
          <CheckCircle2 size={18} className="text-emerald-400 shrink-0" />
          <div>
            <div className="text-sm font-medium text-white">Training is active</div>
            <div className="text-xs text-surface-400">Stop the current run before starting a new one</div>
          </div>
        </div>
      )}
      {submitted && !isRunning && !startError && (
        <div className="glass-card p-4 border-l-4 border-accent-400 flex items-center gap-3">
          <CheckCircle2 size={18} className="text-accent-400 shrink-0" />
          <div>
            <div className="text-sm font-medium text-white">Training job completed</div>
            <div className="text-xs text-surface-400">Configure a new run and launch again</div>
          </div>
        </div>
      )}
      {startError && (
        <div className="glass-card p-4 border-l-4 border-red-500 flex items-center gap-3">
          <AlertCircle size={18} className="text-red-400 shrink-0" />
          <div>
            <div className="text-sm font-medium text-white">Failed to start training</div>
            <div className="text-xs text-red-300 mt-0.5">{startError}</div>
          </div>
        </div>
      )}

      {/* Node Configuration */}
      <div className="glass-card p-6">
        <div className="flex items-center gap-2 mb-5">
          <Network size={16} className="text-accent-400" />
          <span className="text-sm font-semibold text-white">Node Configuration</span>
          <span className="ml-auto text-xs text-surface-500">
            world_size=1 supported for single-machine demo
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Role */}
          <div>
            <label className="block text-xs font-medium text-surface-400 mb-1.5">Role</label>
            <div className="flex gap-2">
              {(['master', 'worker'] as const).map((role) => (
                <button
                  key={role}
                  id={`role-${role}`}
                  onClick={() => update('role', role)}
                  disabled={isRunning}
                  className={`flex-1 py-2.5 px-3 rounded-lg text-sm font-medium border transition-all cursor-pointer ${
                    config.role === role
                      ? 'bg-accent-500/15 border-accent-500/30 text-accent-400'
                      : 'bg-surface-900/50 border-surface-700/50 text-surface-400 hover:border-surface-600'
                  } ${isRunning ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  {role.charAt(0).toUpperCase() + role.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* Rank */}
          <FormField
            label="Rank"
            id="field-rank"
            type="number"
            value={config.rank}
            onChange={(v) => update('rank', parseInt(v) || 0)}
            error={errors.rank}
            disabled={isRunning}
            hint={config.role === 'master' ? 'Master is always rank 0' : undefined}
          />

          {/* World Size */}
          <FormField
            label="World Size"
            id="field-world-size"
            type="number"
            value={config.worldSize}
            onChange={(v) => update('worldSize', parseInt(v) || 1)}
            error={errors.worldSize}
            disabled={isRunning}
            hint="1 = single machine, 2+ = multi-machine DDP"
          />

          {/* Backend */}
          <div>
            <label className="block text-xs font-medium text-surface-400 mb-1.5">Backend</label>
            <div className="flex gap-2">
              {(['nccl', 'gloo'] as const).map((b) => (
                <button
                  key={b}
                  id={`backend-${b}`}
                  onClick={() => update('backend', b)}
                  disabled={isRunning}
                  className={`flex-1 py-2.5 px-3 rounded-lg text-sm font-medium border transition-all cursor-pointer ${
                    config.backend === b
                      ? 'bg-accent-500/15 border-accent-500/30 text-accent-400'
                      : 'bg-surface-900/50 border-surface-700/50 text-surface-400 hover:border-surface-600'
                  } ${isRunning ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  {b.toUpperCase()}
                  <span className="ml-1 text-[10px] opacity-60">
                    {b === 'nccl' ? '(GPU)' : '(CPU)'}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Master Address */}
          <FormField
            label="Master Address"
            id="field-master-addr"
            type="text"
            value={config.masterAddr}
            onChange={(v) => update('masterAddr', v)}
            error={errors.masterAddr}
            disabled={isRunning}
            hint="Tailscale IP of master node (127.0.0.1 for single-machine)"
            placeholder="127.0.0.1"
          />

          {/* Master Port */}
          <FormField
            label="Master Port"
            id="field-master-port"
            type="text"
            value={config.masterPort}
            onChange={(v) => update('masterPort', v)}
            error={errors.masterPort}
            disabled={isRunning}
            placeholder="29500"
          />
        </div>
      </div>

      {/* Training Parameters */}
      <div className="glass-card p-6">
        <div className="flex items-center gap-2 mb-5">
          <Settings size={16} className="text-accent-400" />
          <span className="text-sm font-semibold text-white">Training Parameters</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <FormField label="Epochs"        id="field-epochs"    type="number" value={config.epochs}       onChange={(v) => update('epochs', parseInt(v) || 1)}        error={errors.epochs}       disabled={isRunning} />
          <FormField label="Batch Size"    id="field-batch"     type="number" value={config.batchSize}    onChange={(v) => update('batchSize', parseInt(v) || 64)}    error={errors.batchSize}    disabled={isRunning} />
          <FormField label="Learning Rate" id="field-lr"        type="number" value={config.learningRate} onChange={(v) => update('learningRate', parseFloat(v) || 0.001)} error={errors.learningRate} disabled={isRunning} step="0.0001" />
        </div>

        {/* Init Method */}
        <div className="mt-4">
          <label className="block text-xs font-medium text-surface-400 mb-1.5">Init Method</label>
          <div className="flex gap-2">
            {(['env://', 'tcp://', 'file://'] as const).map((method) => (
              <button
                key={method}
                id={`init-${method}`}
                onClick={() => update('initMethod', method)}
                disabled={isRunning}
                className={`py-2 px-3 rounded-lg text-xs font-mono font-medium border transition-all cursor-pointer ${
                  config.initMethod === method
                    ? 'bg-accent-500/15 border-accent-500/30 text-accent-400'
                    : 'bg-surface-900/50 border-surface-700/50 text-surface-400 hover:border-surface-600'
                } ${isRunning ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                {method}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Generated Command Preview */}
      <div className="glass-card p-5">
        <div className="flex items-center gap-2 mb-3">
          <Cpu size={16} className="text-surface-400" />
          <span className="text-xs font-medium text-surface-400">Generated Command (sent to backend)</span>
        </div>
        <code className="block text-xs font-mono text-surface-300 bg-surface-950/80 p-3 rounded-lg overflow-x-auto whitespace-nowrap">
          {cmd}
        </code>
      </div>

      {/* Action Buttons */}
      <div className="flex flex-wrap gap-3">
        {!isRunning ? (
          <>
            <button
              id="start-training-btn"
              onClick={handleStart}
              disabled={isSubmitting}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 text-white text-sm font-semibold transition-all shadow-lg shadow-blue-500/20 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isSubmitting
                ? <Loader2 size={16} className="animate-spin" />
                : <Play size={16} />
              }
              {isSubmitting ? 'Starting…' : 'Start Training'}
            </button>
            <button
              id="join-cluster-btn"
              onClick={() => update('role', 'worker')}
              disabled={isSubmitting}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-surface-800 hover:bg-surface-700 text-surface-200 text-sm font-medium border border-surface-700/50 transition-all cursor-pointer"
            >
              <UserPlus size={16} />
              Join as Worker
            </button>
          </>
        ) : (
          <button
            id="stop-training-btn"
            onClick={handleStop}
            disabled={isStopping}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-red-500/20 hover:bg-red-500/30 text-red-400 text-sm font-semibold border border-red-500/30 transition-all cursor-pointer disabled:opacity-50"
          >
            {isStopping
              ? <Loader2 size={16} className="animate-spin" />
              : <Square size={16} />
            }
            {isStopping ? 'Stopping…' : 'Stop Training'}
          </button>
        )}
      </div>
    </div>
  );
}

// ── Reusable Form Field ───────────────────────────────────────────────

function FormField({
  label, id, type, value, onChange, error, disabled, hint, placeholder, step,
}: {
  label:       string;
  id:          string;
  type:        string;
  value:       string | number;
  onChange:    (v: string) => void;
  error?:      string;
  disabled?:   boolean;
  hint?:       string;
  placeholder?: string;
  step?:       string;
}) {
  return (
    <div>
      <label htmlFor={id} className="block text-xs font-medium text-surface-400 mb-1.5">
        {label}
      </label>
      <input
        id={id}
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        placeholder={placeholder}
        step={step}
        className={`w-full px-3 py-2.5 rounded-lg text-sm text-white bg-surface-900/50 border transition-all outline-none ${
          error
            ? 'border-red-500/50 focus:border-red-400'
            : 'border-surface-700/50 focus:border-accent-500/50'
        } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
      />
      {error && (
        <div className="flex items-center gap-1 mt-1 text-xs text-red-400">
          <AlertCircle size={12} />
          <span>{error}</span>
        </div>
      )}
      {hint && !error && (
        <div className="text-[11px] text-surface-500 mt-1">{hint}</div>
      )}
    </div>
  );
}
