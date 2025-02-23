import { WS_CONFIG } from './config';
import type { WSRequest, WSResponse } from './types';

class WebSocketManager {
  private ws: WebSocket | null = null;
  private pendingRequests = new Map<string, { 
    resolve: (value: WSResponse) => void;
    reject: (reason: any) => void;
    timeout: number;
  }>();
  private pingInterval: number | null = null;

  constructor() {
    this.connect();
  }

  private connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return;
    
    this.ws = new WebSocket(WS_CONFIG.url);
    
    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      
      // Handle ping messages
      if (message.ping) {
        this.ws!.send(JSON.stringify({ ping: 'pong' }));
        return;
      }

      // Handle normal responses
      const response = message as WSResponse;
      if (response.requestId && this.pendingRequests.has(response.requestId)) {
        const { resolve, timeout } = this.pendingRequests.get(response.requestId)!;
        clearTimeout(timeout);
        this.pendingRequests.delete(response.requestId);
        resolve(response);
      }
    };

    this.ws.onclose = () => {
      if (this.pingInterval) clearInterval(this.pingInterval);
      setTimeout(() => this.connect(), 3000);
    };
  }

  async sendRequest(request: WSRequest): Promise<WSResponse> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected');
    }

    return new Promise((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        this.pendingRequests.delete(request.requestId);
        reject(new Error('Request timeout'));
      }, 30000);

      this.pendingRequests.set(request.requestId, { resolve, reject, timeout });
      this.ws!.send(JSON.stringify(request));
    });
  }
}

export const wsManager = new WebSocketManager(); 