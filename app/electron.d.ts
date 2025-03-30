interface ElectronAPI {
  send: (channel: string, data: any) => void;
  receive: (channel: string, callback: Function) => void;
}

declare global {
  interface Window {
    electron: ElectronAPI;
  }
}
