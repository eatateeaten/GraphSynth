import { useRef, useEffect } from 'react';
import { Workspace, type WorkspaceHandle } from './Workspace';
import { Sidebar } from './Sidebar';
import { Topbar } from './Topbar';
import type { Layer, WSResponse } from './types';

function App() {
  const ws = useRef<WebSocket>();
  const workspaceRef = useRef<WorkspaceHandle>(null);

  useEffect(() => {
    // Initialize WebSocket connection
    ws.current = new WebSocket('ws://localhost:8765');
    ws.current.onmessage = (event) => {
      const response: WSResponse = JSON.parse(event.data);
      console.log(response);
    };

    return () => ws.current?.close();
  }, []);

  const handleLayerAdd = (layer: Layer) => {
    workspaceRef.current?.addLayer(layer);
  };

  return (
    <div className="app">
      <div className="topbar-container">
        <Topbar />
      </div>
      <main>
        <div className="workspace-container">
          <Workspace ref={workspaceRef} />
        </div>
        <div className="sidebar-container">
          <Sidebar onAddLayer={handleLayerAdd} />
        </div>
      </main>
    </div>
  );
}

export default App;
