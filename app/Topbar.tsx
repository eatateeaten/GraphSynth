import { Menu, Button } from '@mantine/core';
import { JupyterButton } from './JupyterButton';

export function Topbar() {
    return (
        <div style={{ padding: '8px 16px', display: 'flex', gap: '4px', alignItems: 'center' }}>
            <Menu shadow="md">
                <Menu.Target>
                    <Button variant="subtle">File</Button>
                </Menu.Target>

                <Menu.Dropdown>
                    <Menu.Item>Save</Menu.Item>
                    <Menu.Item>Load</Menu.Item>
                    <Menu.Divider />
                    <Menu.Item>New</Menu.Item>
                </Menu.Dropdown>
            </Menu>

            <Menu shadow="md">
                <Menu.Target>
                    <Button variant="subtle">Edit</Button>
                </Menu.Target>

                <Menu.Dropdown>
                    <Menu.Item>Copy</Menu.Item>
                    <Menu.Item>Cut</Menu.Item>
                    <Menu.Item>Paste</Menu.Item>
                </Menu.Dropdown>
            </Menu>

            <Menu shadow="md">
                <Menu.Target>
                    <Button variant="subtle">View</Button>
                </Menu.Target>

                <Menu.Dropdown>
                    <Menu.Item>Zoom In</Menu.Item>
                    <Menu.Item>Zoom Out</Menu.Item>
                    <Menu.Item>Reset Zoom</Menu.Item>
                </Menu.Dropdown>
            </Menu>

            <Menu shadow="md">
                <Menu.Target>
                    <Button variant="subtle">Help</Button>
                </Menu.Target>

                <Menu.Dropdown>
                    <Menu.Item>About</Menu.Item>
                </Menu.Dropdown>
            </Menu>
            
            {/* Spacer to push Jupyter connection to the right */}
            <div style={{ flex: 1 }}></div>
            
            {/* Jupyter connection panel */}
            <JupyterButton />
        </div>
    );
} 
