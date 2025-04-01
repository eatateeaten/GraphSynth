import { Workspace } from './Workspace';
import { NodeEditor } from './NodeEditor';
import { CodeWindow } from './CodeWindow';
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
                    <div className="node-editor-container"><NodeEditor /></div>
                    <div className="code-window-container"><CodeWindow /></div>
                </div>
            </main>
        </div>
    );
}

export default App;
