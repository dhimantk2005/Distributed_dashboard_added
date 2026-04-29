import { useState } from 'react';
import { useBackend } from './hooks/useBackend';
import Sidebar from './components/layout/Sidebar';
import ClusterView from './components/cluster/ClusterView';
import TrainingDashboard from './components/metrics/TrainingDashboard';
import ControlPanel from './components/control/ControlPanel';
import LogsPanel from './components/logs/LogsPanel';
import DiagnosticsPanel from './components/diagnostics/DiagnosticsPanel';

export default function App() {
  const [currentPage, setCurrentPage] = useState('cluster');
  const ws = useBackend();

  const renderPage = () => {
    switch (currentPage) {
      case 'cluster':
        return <ClusterView nodes={ws.nodes} />;
      case 'training':
        return <TrainingDashboard training={ws.training} />;
      case 'control':
        return (
          <ControlPanel
            isRunning={ws.isRunning}
            onStart={ws.startTraining}
            onStop={ws.stopTraining}
          />
        );
      case 'logs':
        return <LogsPanel logs={ws.logs} terminalLines={ws.terminalLines} />;
      case 'diagnostics':
        return (
          <DiagnosticsPanel
            onRun={ws.runDiagnostics}
            result={ws.diagnostics}
            isRunning={ws.isDiagnosticsRunning}
          />
        );
      default:
        return <ClusterView nodes={ws.nodes} />;
    }
  };

  return (
    <div className="flex min-h-screen">
      <Sidebar
        currentPage={currentPage}
        onPageChange={setCurrentPage}
        isConnected={ws.isConnected}
        isRunning={ws.isRunning}
        connectionError={ws.connectionError}
      />

      {/* Main Content */}
      <main
        className="flex-1 ml-[240px] transition-all duration-300"
        style={{ minHeight: '100vh' }}
      >
        <div className="p-6 lg:p-8 max-w-[1440px] mx-auto">
          {renderPage()}
        </div>
      </main>
    </div>
  );
}
