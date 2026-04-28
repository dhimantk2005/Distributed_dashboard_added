import { useState } from 'react';
import {
  LayoutGrid,
  Activity,
  Settings,
  Terminal,
  Cpu,
  Wifi,
  WifiOff,
  AlertTriangle,
} from 'lucide-react';

interface SidebarProps {
  currentPage:     string;
  onPageChange:    (page: string) => void;
  isConnected:     boolean;
  isRunning:       boolean;
  connectionError: string | null;
}

const navItems = [
  { id: 'cluster',     label: 'Cluster',     icon: LayoutGrid },
  { id: 'training',    label: 'Training',    icon: Activity   },
  { id: 'control',     label: 'Control',     icon: Settings   },
  { id: 'logs',        label: 'Logs',        icon: Terminal   },
  { id: 'diagnostics', label: 'Diagnostics', icon: Cpu        },
];

export default function Sidebar({
  currentPage,
  onPageChange,
  isConnected,
  isRunning,
  connectionError,
}: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={`fixed left-0 top-0 h-full z-50 flex flex-col transition-all duration-300 ease-in-out ${
        collapsed ? 'w-[68px]' : 'w-[240px]'
      }`}
      style={{
        background: 'linear-gradient(180deg, rgba(15,17,23,0.98) 0%, rgba(10,12,18,0.98) 100%)',
        borderRight: '1px solid rgba(255,255,255,0.06)',
      }}
    >
      {/* Header */}
      <div
        className="flex items-center gap-3 px-4 h-16 cursor-pointer select-none"
        onClick={() => setCollapsed(!collapsed)}
      >
        <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shrink-0">
          <Cpu size={20} className="text-white" />
        </div>
        {!collapsed && (
          <div className="animate-fade-in overflow-hidden">
            <div className="text-sm font-semibold text-white tracking-tight leading-tight">
              DDP Control
            </div>
            <div className="text-[10px] text-surface-400 font-medium uppercase tracking-widest">
              Distributed Training
            </div>
          </div>
        )}
      </div>

      {/* Divider */}
      <div className="mx-3 h-px bg-white/5 mb-2" />

      {/* Navigation */}
      <nav className="flex-1 px-2 space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentPage === item.id;
          return (
            <button
              key={item.id}
              id={`nav-${item.id}`}
              onClick={() => onPageChange(item.id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 cursor-pointer ${
                isActive
                  ? 'bg-accent-500/15 text-accent-400'
                  : 'text-surface-400 hover:text-surface-200 hover:bg-white/5'
              }`}
            >
              <Icon size={18} className="shrink-0" />
              {!collapsed && <span className="animate-fade-in">{item.label}</span>}
            </button>
          );
        })}
      </nav>

      {/* Footer Status */}
      <div className="px-3 pb-4 space-y-2">
        <div className="mx-1 h-px bg-white/5 mb-3" />

        {/* Connection Status */}
        <div className="flex items-center gap-2 px-2 py-1.5">
          {isConnected ? (
            <Wifi size={14} className="text-emerald-400 shrink-0" />
          ) : connectionError ? (
            <AlertTriangle size={14} className="text-amber-400 shrink-0" />
          ) : (
            <WifiOff size={14} className="text-red-400 shrink-0" />
          )}
          {!collapsed && (
            <span
              className={`text-xs font-medium animate-fade-in truncate ${
                isConnected
                  ? 'text-emerald-400'
                  : connectionError
                  ? 'text-amber-400'
                  : 'text-red-400'
              }`}
              title={connectionError ?? undefined}
            >
              {isConnected
                ? 'Backend Connected'
                : connectionError
                ? 'Backend Offline'
                : 'Connecting…'}
            </span>
          )}
        </div>

        {/* Training Status */}
        <div className="flex items-center gap-2 px-2 py-1.5">
          <div className="status-dot-wrapper shrink-0">
            <div
              className={`w-2 h-2 rounded-full ${isRunning ? 'bg-emerald-400' : 'bg-surface-500'}`}
            />
            {isRunning && <div className="status-dot-ping bg-emerald-400/40" />}
          </div>
          {!collapsed && (
            <span
              className={`text-xs font-medium animate-fade-in ${
                isRunning ? 'text-emerald-400' : 'text-surface-500'
              }`}
            >
              {isRunning ? 'Training Active' : 'Idle'}
            </span>
          )}
        </div>
      </div>
    </aside>
  );
}
