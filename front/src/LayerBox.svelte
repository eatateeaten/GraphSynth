<script lang="ts">
  interface Layer {
    name: string;
    type: string;
    params: Record<string, string | number | number[]>;
    valid: boolean;
    inDim?: string;
    outDim?: string;
    input_nodes?: string[];
    output_nodes?: string[];
  }
  
  export let layer: Layer;
</script>

<div class="layer-box" class:invalid={!layer.valid}>
  <div class="layer-title">{layer.type}</div>
  {#if Object.keys(layer.params).length > 0}
    <div class="layer-params">
      {#each Object.entries(layer.params) as [key, value]}
        <div class="param">{key}={Array.isArray(value) ? `[${value.join(',')}]` : value}</div>
      {/each}
    </div>
  {/if}
  {#if layer.inDim}
    <div class="dim in-dim">in: {layer.inDim}</div>
  {/if}
  {#if layer.outDim}
    <div class="dim out-dim">out: {layer.outDim}</div>
  {/if}
</div>

<style>
  .layer-box {
    background: #2d2d2d;
    border: 1px solid #404040;
    border-radius: 4px;
    padding: 1rem;
    width: 200px;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    flex-shrink: 0;
  }

  .layer-box.invalid {
    border-color: #f44336;
  }

  .layer-title {
    color: #d4d4d4;
    font-weight: bold;
    text-align: center;
    font-family: 'Consolas', 'Monaco', monospace;
  }

  .layer-params {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    color: #9cdcfe;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 0.9em;
  }

  .dim {
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 0.9em;
    padding: 0.25rem;
    border-radius: 2px;
  }

  .in-dim {
    color: #4ec9b0;
  }

  .out-dim {
    color: #569cd6;
  }
</style>
