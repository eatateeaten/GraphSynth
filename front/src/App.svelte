<script lang="ts">
  import { onMount } from 'svelte';
  import LayerBox from './LayerBox.svelte';
  import './style.css';

  type LayerParams = Record<string, string | number | number[]>;

  interface Layer {
    name: string;
    type: string;
    params: LayerParams;
    valid: boolean;
    inDim?: string;
    outDim?: string;
    input_nodes?: string[];
    output_nodes?: string[];
  }

  let input = '';
  let layers: Layer[] = [];
  let parseError = '';
  let ws: WebSocket;
  let pythonCode = '';

  // Initialize WebSocket connection
  function initWebSocket() {
    ws = new WebSocket('ws://localhost:8765');
    ws.onmessage = (event) => {
      const response = JSON.parse(event.data);
      if (response.success) {
        pythonCode = response.pytorch_code;
        parseError = '';
      } else {
        parseError = response.error;
      }
    };
  }

  // Connect on component mount
  onMount(() => {
    initWebSocket();
    return () => ws?.close();
  });

  function computeDimensions(layer: Layer): void {
    if (layer.type === 'conv2d') {
      layer.inDim = `(B, ${layer.params.in_channels}, H, W)`;
      layer.outDim = `(B, ${layer.params.out_channels}, H', W')`;
    } else if (layer.type === 'batchnorm') {
      layer.inDim = `(B, ${layer.params.num_features}, H, W)`;
      layer.outDim = layer.inDim;
    } else if (layer.type === 'relu' || layer.type === 'maxpool') {
      layer.inDim = '(B, C, H, W)';
      layer.outDim = '(B, C, H\', W\')';
    } else if (layer.type === 'linear') {
      layer.inDim = `(B, ${layer.params.in_features})`;
      layer.outDim = `(B, ${layer.params.out_features})`;
    }
  }

  function parseLayer(line: string, index: number): Layer {
    line = line.trim();
    if (!line) return { name: '', type: '', params: {}, valid: false };

    // Handle parameterless layers like 'relu'
    if (!line.includes('(')) {
      const layer = { 
        name: `${line}_${index}`,
        type: line, 
        params: {}, 
        valid: true,
        input_nodes: [],
        output_nodes: []
      };
      computeDimensions(layer);
      return layer;
    }

    // Parse "type(param1=value1, param2=value2)" format
    const match = line.match(/^(\w+)\((.*)\)$/);
    if (!match) {
      return { name: '', type: line, params: {}, valid: false };
    }

    const [_, type, paramsStr] = match;
    const params: LayerParams = {};
    
    if (paramsStr.trim()) {
      paramsStr.split(',').forEach(param => {
        const [key, value] = param.trim().split('=');
        if (key && value) {
          // Handle array parameters (kernel_size, stride, padding)
          if (value.includes('[')) {
            params[key.trim()] = JSON.parse(value.trim());
          } else {
            params[key.trim()] = isNaN(Number(value)) ? value.trim() : Number(value);
          }
        }
      });
    }

    // Convert parameter names to match schema
    const paramMap: Record<string, string> = {
      'in': 'in_channels',
      'out': 'out_channels',
      'features': 'num_features',
      'p': 'probability'
    };

    const convertedParams = Object.entries(params).reduce((acc, [key, value]) => {
      acc[paramMap[key] || key] = value;
      return acc;
    }, {} as LayerParams);

    const layer = { 
      name: `${type}_${index}`,
      type, 
      params: convertedParams, 
      valid: true,
      input_nodes: [],
      output_nodes: []
    };
    computeDimensions(layer);
    return layer;
  }

  function updateNetwork(text: string): void {
    try {
      // Parse layers from text
      const newLayers = text.split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 0)
        .map((line, index) => parseLayer(line, index));

      // Set up connections
      newLayers.forEach((layer, i) => {
        if (i > 0) {
          layer.input_nodes = [newLayers[i - 1].name];
        }
        if (i < newLayers.length - 1) {
          layer.output_nodes = [newLayers[i + 1].name];
        }
      });

      layers = newLayers;
      parseError = '';

      // Convert to JSON format and send to server
      if (ws?.readyState === WebSocket.OPEN) {
        const graphData = {
          name: 'network',
          nodes: layers.map(({ name, type, params, input_nodes, output_nodes }) => ({
            name,
            type,
            params,
            input_nodes: input_nodes || [],
            output_nodes: output_nodes || []
          }))
        };
        ws.send(JSON.stringify(graphData));
      }
    } catch (e: unknown) {
      parseError = e instanceof Error ? e.message : String(e);
      layers = [];
    }
  }

  $: updateNetwork(input);
</script>

<main>
  <div class="flowchart">
    {#each layers as layer, i}
      <div class="layer-container">
        <LayerBox {layer} />
        {#if i < layers.length - 1}
          <div class="arrow">→</div>
        {/if}
      </div>
    {/each}
  </div>
  
  <div class="input">
    <textarea 
      bind:value={input} 
      placeholder="Enter your network definition here...
Example:
conv2d(in=3, out=16, kernel=[3,3])
batchnorm(features=16)
relu
maxpool(kernel=[2,2])
linear(in=1024, out=10)"
      rows="4"
    ></textarea>
    {#if parseError}
      <div class="error">{parseError}</div>
    {/if}
    {#if pythonCode}
      <pre class="code">{pythonCode}</pre>
    {/if}
  </div>
</main>

<style>
  .code {
    margin-top: 1rem;
    padding: 1rem;
    background: #1e1e1e;
    border-radius: 4px;
    color: #d4d4d4;
    font-family: 'Consolas', 'Monaco', monospace;
    white-space: pre-wrap;
  }
</style>
