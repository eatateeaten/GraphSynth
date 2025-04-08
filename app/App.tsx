import { Workspace } from './Workspace';
import { NodeEditor } from './NodeEditor';
import { CodeWindow } from './CodeWindow';
import { CodeOutput } from './CodeOutput';
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
                </div>
            </main>
            <div className="bottom-container">
                <div className="code-window-container"><CodeWindow /></div>
                <div className="code-output-container"><CodeOutput /></div>
            </div>
        </div>
    );
}

export default App;
