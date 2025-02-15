import asyncio
import json
import uuid
import websockets
from jsonschema import validate
from network_seq import (Node, Conv1DNode, Conv2DNode, Conv3DNode, 
                        ElementWiseNonlinearity, ElementWiseNonlinearityType,
                        LinearNode, FlattenNode, Seq)

# Load JSON schema
with open('schema.json', 'r') as f:
    SCHEMA = json.load(f)



def create_node_from_json(node_data):
    """Create a Node instance from JSON data."""
    node_type = node_data['type']
    params = node_data['params']
    
    if node_type == 'conv1d':
        return Conv1DNode(
            batch_size=params['batch_size'],
            in_channels=params['in_channels'],
            out_channels=params['out_channels'],
            input_size=params['input_size'],
            kernel_size=params['kernel_size'],
            stride=params.get('stride', 1),
            padding=params.get('padding', 0)
        )
    elif node_type == 'conv2d':
        return Conv2DNode(
            batch_size=params['batch_size'],
            in_channels=params['in_channels'],
            out_channels=params['out_channels'],
            input_size=tuple(params['input_size']),
            kernel_size=params['kernel_size'],
            stride=params.get('stride', 1),
            padding=params.get('padding', 0)
        )
    elif node_type == 'elementwise_nonlinearity':
        return ElementWiseNonlinearity(
            dim=tuple(params['dim']),
            nonlinearity=ElementWiseNonlinearityType[params['nonlinearity'].upper()]
        )
    elif node_type == 'linear':
        return LinearNode(
            batch_size=params['batch_size'],
            input_features=params['input_features'],
            output_features=params['output_features']
        )
    elif node_type == 'flatten':
        return FlattenNode(
            dim=tuple(params['dim']),
            start_dim=params.get('start_dim', 1),
            end_dim=params.get('end_dim', -1)
        )
    else:
        raise ValueError(f"Unknown node type: {node_type}")

def create_sequence_from_json(data):
    """Create a Seq instance from JSON data."""
    # Create nodes in order
    nodes = []
    node_map = {}
    
    # First pass: create all nodes
    for node_data in data['nodes']:
        node = create_node_from_json(node_data)
        node_map[node_data['name']] = node
        nodes.append(node)
    
    # Create sequence
    try:
        seq = Seq(nodes)
        return seq
    except Exception as e:
        raise ValueError(f"Failed to create sequence: {str(e)}")

async def handle_message(websocket):
    """Handle incoming WebSocket messages with different operations."""
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                
                # Check if operation field exists
                if 'operation' not in data:
                    await websocket.send(json.dumps({
                        'success': False,
                        'error': 'Missing operation field'
                    }))
                    continue

                # Handle different operations
                if data['operation'] == 'addNode':
                    if 'layer' not in data:
                        await websocket.send(json.dumps({
                            'success': False,
                            'error': 'Missing layer data for addNode operation'
                        }))
                        continue
                    
                    try:
                        # Validate against schema
                        validate(instance=data['layer'], schema=SCHEMA)
                        
                        # Process the node
                        node = create_node_from_json(data['layer'])
                        response = {
                            'success': True,
                            'id': str(uuid.uuid4()),
                            'operation': 'addNode',
                            'pytorch_code': node.to_pytorch_code()
                        }
                    except Exception as e:
                        response = {
                            'success': False,
                            'operation': 'addNode',
                            'error': str(e)
                        }
                    
                    await websocket.send(json.dumps(response))

                else:
                    await websocket.send(json.dumps({
                        'success': False,
                        'error': f'Unknown operation: {data["operation"]}'
                    }))

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
    async with websockets.serve(handle_message, "localhost", 8765):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
