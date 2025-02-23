// Grid configuration
export const GRID_SIZE = 10;

// WebSocket configuration
export const WS_CONFIG = {
  url: 'ws://localhost:8765',
  options: {
    share: true,
    shouldReconnect: (closeEvent: CloseEvent) => true,
    reconnectInterval: 3000,
    reconnectAttempts: 10,
    heartbeat: {
      message: JSON.stringify({ type: 'ping' }),
      returnMessage: JSON.stringify({ type: 'pong' }),
      timeout: 60000,    // 1 minute
      interval: 25000    // 25 seconds
    }
  }
} as const; 