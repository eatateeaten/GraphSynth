import { Workspace } from './Workspace';
import { Sidebar } from './Sidebar';
import { Topbar } from './Topbar';

function App() {
  return (
    <div className="app">
      <div className="topbar-container">
        <Topbar />
      </div>
      <main>
        <div className="workspace-container">
          <Workspace />
        </div>
        <div className="sidebar-container">
          <Sidebar />
        </div>
      </main>
    </div>
  );
}

export default App;
