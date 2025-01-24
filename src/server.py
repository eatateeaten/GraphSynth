import asyncio
import json
import uuid
from pathlib import Path
import websockets
from jsonschema import validate
from network_graph import Node, ConvNode, BatchNormNode, ReLUNode, MaxPoolNode, DropoutNode, LinearNode, Graph

# Load JSON schema
with open('schema/schema.json', 'r') as f:
    SCHEMA = json.load(f)

def create_node_from_json(node_data):
    """Create a Node instance from JSON data."""
    node_type = node_data['type']
    params = node_data['params']
    
    if node_type == 'conv2d':
        return ConvNode(
            batch_size=1,  # Default batch size
            in_channels=params['in_channels'],
            out_channels=params['out_channels'],
            input_size=(224, 224),  # Default input size
            kernel_size=params['kernel_size'],
            stride=params.get('stride', (1, 1)),
            padding=params.get('padding', (0, 0))
        )
    elif node_type == 'batchnorm':
        return BatchNormNode(num_features=params['num_features'])
    elif node_type == 'relu':
        return ReLUNode(dim=(params.get('features', 1),))
    elif node_type == 'maxpool':
        return MaxPoolNode(
            kernel_size=params['kernel_size'],
            stride=params.get('stride'),
            padding=params.get('padding', 0)
        )
    elif node_type == 'dropout':
        return DropoutNode(p=params['probability'])
    elif node_type == 'linear':
        return LinearNode(
            in_features=params['in_features'],
            out_features=params['out_features']
        )
    else:
        raise ValueError(f"Unknown node type: {node_type}")

def create_graph_from_json(data):
    """Create a Graph instance from JSON data."""
    # First create all nodes
    nodes = {}
    for node_data in data['nodes']:
        node = create_node_from_json(node_data)
        node.name = node_data['name']  # Override auto-generated name
        nodes[node.name] = node
    
    # Then connect them
    for node_data in data['nodes']:
        node = nodes[node_data['name']]
        for out_name in node_data.get('output_nodes', []):
            if out_name in nodes:
                node.connect_to(nodes[out_name])
    
    # Find input and output nodes
    input_nodes = [n for n in nodes.values() if not n.input_nodes]
    output_nodes = [n for n in nodes.values() if not n.output_nodes]
    
    if not input_nodes or not output_nodes:
        raise ValueError("Graph must have at least one input and one output node")
    
    # Create graph with first input and output node
    graph = Graph(input_nodes[0], output_nodes[0])
    
    # Add all other nodes
    for node in nodes.values():
        if node not in [input_nodes[0], output_nodes[0]]:
            graph.add_node(node)
    
    return graph

async def handle_graph(websocket):
    """Handle incoming WebSocket connections and messages."""
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                
                # Validate against schema
                validate(instance=data, schema=SCHEMA)

                # Process the graph
                try:
                    graph = create_graph_from_json(data)
                    response = {
                        'success': True,
                        'graph_id': str(uuid.uuid4()),
                        'pytorch_code': graph.to_pytorch_code()
                    }
                except Exception as e:
                    response = {
                        'success': False,
                        'error': str(e)
                    }

                # Send response
                await websocket.send(json.dumps(response))
                
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    'success': False,
                    'error': 'Invalid JSON'
                }))
            except Exception as e:
                await websocket.send(json.dumps({
                    'success': False,
                    'error': str(e)
                }))

    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        print(f"Error handling message: {e}")

async def main():
    """Start the WebSocket server."""
    async with websockets.serve(handle_graph, "localhost", 8765):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
