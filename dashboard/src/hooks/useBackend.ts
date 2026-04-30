/**
 * useBackend — real WebSocket + REST client.
 *
 * Replaces useWebSocket.ts.  Connects to the FastAPI backend at
 * localhost:8000, receives cluster state / log / metric events over
 * WebSocket, and exposes REST actions for starting training and
 * running diagnostics.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import type {
  ClusterNode,
  TrainingState,
  LogEntry,
  TerminalLine,
  TrainingConfig,
  DiagnosticsResult,
} from '../types';

// ── Constants ─────────────────────────────────────────────────────────

const BACKEND_HTTP = 'http://localhost:8000';
const BACKEND_WS   = 'ws://localhost:8000/ws';
const MAX_LOGS     = 500;
const MAX_TERMINAL = 1500;

// ── Helpers ───────────────────────────────────────────────────────────

function makeInitialTraining(): TrainingState {
  return {
    isRunning:      false,
    currentEpoch:   0,
    totalEpochs:    0,
    globalLoss:     0,
    globalAccuracy: 0,
    elapsedTime:    0,
    epochHistory:   [],
    perNodeMetrics: [],
  };
}

// ── Public interface ──────────────────────────────────────────────────

export interface StartTrainingResult {
  success: boolean;
  error?: string;
  job_id?: string;
}

export interface UseBackendReturn {
  nodes:               ClusterNode[];
  training:            TrainingState;
  logs:                LogEntry[];
  terminalLines:       TerminalLine[];
  isConnected:         boolean;
  isRunning:           boolean;
  connectionError:     string | null;
  startTraining:       (config: TrainingConfig) => Promise<StartTrainingResult>;
  stopTraining:        () => Promise<void>;
  resetCluster:        () => void;
  runDiagnostics:      () => Promise<DiagnosticsResult>;
  diagnostics:         DiagnosticsResult | null;
  isDiagnosticsRunning:boolean;
}

// ── Hook ──────────────────────────────────────────────────────────────

export function useBackend(): UseBackendReturn {
  const [nodes,               setNodes]               = useState<ClusterNode[]>([]);
  const [training,            setTraining]            = useState<TrainingState>(makeInitialTraining);
  const [logs,                setLogs]                = useState<LogEntry[]>([]);
  const [terminalLines,       setTerminalLines]       = useState<TerminalLine[]>([]);
  const [isConnected,         setIsConnected]         = useState(false);
  const [isRunning,           setIsRunning]           = useState(false);
  const [connectionError,     setConnectionError]     = useState<string | null>(null);
  const [diagnostics,         setDiagnostics]         = useState<DiagnosticsResult | null>(null);
  const [isDiagnosticsRunning,setIsDiagnosticsRunning]= useState(false);

  const wsRef            = useRef<WebSocket | null>(null);
  const reconnectTimer   = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectDelay   = useRef(1000);   // ms, grows up to 10 s
  const isMounted        = useRef(true);

  // ── Message handler ─────────────────────────────────────────────────
  // Store in a ref so the stable `connect` callback always dispatches to
  // the latest version (avoids stale closure over state setters).

  const stateSetters = useRef({
    setNodes, setTraining, setLogs, setTerminalLines,
    setIsRunning, setDiagnostics, setIsDiagnosticsRunning,
  });
  stateSetters.current = {
    setNodes, setTraining, setLogs, setTerminalLines,
    setIsRunning, setDiagnostics, setIsDiagnosticsRunning,
  };

  const handleMessage = useCallback((raw: string) => {
    let msg: Record<string, unknown>;
    try { msg = JSON.parse(raw); }
    catch { return; }

    const {
      setNodes, setTraining, setLogs, setTerminalLines,
      setIsRunning, setDiagnostics, setIsDiagnosticsRunning,
    } = stateSetters.current;

    switch (msg.type as string) {

      case 'init': {
        const n  = (msg.nodes    as ClusterNode[] | undefined)  ?? [];
        const t  = (msg.training as TrainingState | undefined)  ?? makeInitialTraining();
        const l  = (msg.logs     as LogEntry[]    | undefined)  ?? [];
        const tl = (msg.terminalLines as TerminalLine[] | undefined) ?? [];
        setNodes(n);
        setTraining(t);
        setLogs(l.slice(-MAX_LOGS));
        setTerminalLines(tl.slice(-MAX_TERMINAL));
        setIsRunning(t.isRunning);
        break;
      }

      case 'cluster_state': {
        const n = (msg.nodes    as ClusterNode[] | undefined) ?? [];
        const t = (msg.training as TrainingState | undefined) ?? makeInitialTraining();
        setNodes(n);
        setTraining(t);
        setIsRunning(t.isRunning);
        break;
      }

      case 'log': {
        const entry = msg.entry as LogEntry | undefined;
        if (entry) {
          setLogs((prev) => [...prev, entry].slice(-MAX_LOGS));
        }
        break;
      }

      case 'terminal': {
        const entry = msg.entry as TerminalLine | undefined;
        if (entry) {
          setTerminalLines((prev) => [...prev, entry].slice(-MAX_TERMINAL));
        }
        break;
      }

      case 'training_metrics': {
        const t = msg.training as TrainingState | undefined;
        if (t) {
          setTraining(t);
          setIsRunning(t.isRunning);
        }
        break;
      }

      case 'job_complete': {
        setIsRunning(false);
        const n = msg.nodes    as ClusterNode[] | undefined;
        const t = msg.training as TrainingState | undefined;
        if (n) setNodes(n);
        if (t) setTraining(t);
        break;
      }

      case 'diagnostics_result': {
        const r = msg.result as DiagnosticsResult | undefined;
        if (r) {
          setDiagnostics(r);
          setIsDiagnosticsRunning(false);
        }
        break;
      }

      default:
        break; // heartbeat etc.
    }
  }, []); // stable — reads state via ref

  // ── Connection management ────────────────────────────────────────────

  const connect = useCallback(() => {
    if (!isMounted.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    let ws: WebSocket;
    try {
      ws = new WebSocket(BACKEND_WS);
    } catch (e) {
      setConnectionError(`Failed to create WebSocket: ${e}`);
      return;
    }

    wsRef.current = ws;

    ws.onopen = () => {
      if (!isMounted.current) { ws.close(); return; }
      setIsConnected(true);
      setConnectionError(null);
      reconnectDelay.current = 1000;
    };

    ws.onmessage = (ev) => handleMessage(ev.data as string);

    ws.onclose = () => {
      if (!isMounted.current) return;
      setIsConnected(false);
      // Exponential backoff reconnect
      const delay = reconnectDelay.current;
      reconnectDelay.current = Math.min(delay * 1.5, 10_000);
      reconnectTimer.current = setTimeout(connect, delay);
    };

    ws.onerror = () => {
      setConnectionError(`Cannot reach backend at ${BACKEND_WS}. Is it running?`);
      ws.close();
    };
  }, [handleMessage]);

  useEffect(() => {
    isMounted.current = true;
    connect();
    // Ping every 20 s to keep the connection alive
    const pingInterval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send('ping');
      }
    }, 20_000);

    return () => {
      isMounted.current = false;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      clearInterval(pingInterval);
      wsRef.current?.close();
    };
  }, [connect]);

  // ── Actions ──────────────────────────────────────────────────────────

  const startTraining = useCallback(async (config: TrainingConfig): Promise<StartTrainingResult> => {
    const res = await fetch(`${BACKEND_HTTP}/api/jobs/start`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(config),
    });
    const data: StartTrainingResult = await res.json();
    if (data.success) setIsRunning(true);
    return data;
  }, []);

  const stopTraining = useCallback(async () => {
    await fetch(`${BACKEND_HTTP}/api/jobs/stop`, { method: 'POST' });
    setIsRunning(false);
  }, []);

  const resetCluster = useCallback(() => {
    setNodes([]);
    setTraining(makeInitialTraining());
    setLogs([]);
    setIsRunning(false);
  }, []);

  const runDiagnostics = useCallback(async (): Promise<DiagnosticsResult> => {
    setIsDiagnosticsRunning(true);
    const res  = await fetch(`${BACKEND_HTTP}/api/diagnostics/run`, { method: 'POST' });
    const data = await res.json() as DiagnosticsResult;
    setDiagnostics(data);
    setIsDiagnosticsRunning(false);
    return data;
  }, []);

  // ── Return ────────────────────────────────────────────────────────────

  return {
    nodes,
    training,
    logs,
    terminalLines,
    isConnected,
    isRunning,
    connectionError,
    startTraining,
    stopTraining,
    resetCluster,
    runDiagnostics,
    diagnostics,
    isDiagnosticsRunning,
  };
}
