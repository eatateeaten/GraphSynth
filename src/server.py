import asyncio
import json
import logging
import structlog
import websockets
from typing import Dict, Any, Tuple
from node_v3 import Node, Tensor, Reshape
import numpy as np

# Configure structured logging
logging.basicConfig(level=logging.INFO)
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer()
    ],
)

logger = structlog.get_logger()

# Store active connections and nodes
active_connections: Dict[int, websockets.WebSocketServerProtocol] = {}
active_nodes: Dict[str, Node] = {}  # Store nodes by UUID
connection_counter = 0

def create_node(node_type: str, params: Dict[str, Any]) -> Node:
    """Create a node of the specified type with given parameters."""
    # Convert to lowercase for comparison
    node_type = node_type.lower()
    if node_type == "tensor":
        # For tensors, we expect data and shape in the params
        data = params.get('data')
        shape = params.get('shape')
        if not data or not shape:
            raise ValueError("Tensor requires both data and shape parameters")
        # Convert data to numpy array with the right shape
        data = np.array(data).reshape(shape)
        return Tensor(data)
    elif node_type == "reshape":
        # For reshape, we expect out_dim in the params
        out_dim = params.get('out_dim')
        if not out_dim:
            raise ValueError("Reshape requires out_dim parameter")
        return Reshape(tuple(out_dim))
    else:
        raise ValueError(f"Unsupported node type: {node_type}")

def handle_add_node(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new node and store it."""
    node_type = data.get('type')
    params = data.get('params', {})
    if not node_type:
        raise ValueError("Node type not specified")
        
    node = create_node(node_type, params)
    active_nodes[node.id] = node
    
    return {
        "success": True,
        "requestId": data.get('requestId'),
        "id": node.id,
        "in_shape": node.in_shape,
        "out_shape": node.out_shape,
        "completed": node.completed()
    }

def get_nodes_for_connection(node_id: str, other_id: str) -> Tuple[Node, Node]:
    """
    Helper function to get two nodes for connection operations.
    Raises ValueError if either node is not found.
    """
    if not node_id or not other_id:
        raise ValueError("Both node IDs required")
        
    node = active_nodes.get(node_id)
    other_node = active_nodes.get(other_id)
    if not node or not other_node:
        raise ValueError("Node not found")
        
    return node, other_node

def handle_set_input_node(data: Dict[str, Any]) -> Dict[str, Any]:
    """Connect input node and validate shapes."""
    node, input_node = get_nodes_for_connection(
        data.get('nodeId'),
        data.get('inputId')
    )
    node.set_input_node(input_node)
    return {
        "success": True,
        "requestId": data.get('requestId'),
        "id": node.id,
        "in_shape": node.in_shape,
        "out_shape": node.out_shape,
        "completed": node.completed()
    }

def handle_set_output_node(data: Dict[str, Any]) -> Dict[str, Any]:
    """Connect output node and validate shapes."""
    node, output_node = get_nodes_for_connection(
        data.get('nodeId'),
        data.get('outputId')
    )
    node.set_output_node(output_node)
    return {
        "success": True,
        "requestId": data.get('requestId'),
        "id": node.id,
        "in_shape": node.in_shape,
        "out_shape": node.out_shape,
        "completed": node.completed()
    }

async def handle_message(websocket: websockets.WebSocketServerProtocol, connection_id: int, data: Dict[str, Any]):
    """Handle incoming websocket messages"""
    try:
        # Handle pong responses
        if 'ping' in data:
            return

        operation = data.get('operation')
        if not operation:
            raise ValueError("No operation specified in message")
            
        # Map operations to their handlers
        handlers = {
            "addNode": handle_add_node,
            "setInputNode": handle_set_input_node,
            "setOutputNode": handle_set_output_node
        }
        
        handler = handlers.get(operation)
        if not handler:
            raise ValueError(f"Unknown operation: {operation}")
            
        response = handler(data)
        await websocket.send(json.dumps(response))
        logger.info("message_processed",
                   connection_id=connection_id,
                   operation=operation)
                   
    except Exception as e:
        error_msg = str(e)
        logger.error("message_processing_error",
                    connection_id=connection_id,
                    error=error_msg)
        await websocket.send(json.dumps({
            "success": False,
            "requestId": data.get('requestId'),
            "error": error_msg
        }))

async def send_ping(websocket: websockets.WebSocketServerProtocol, connection_id: int):
    """Send periodic pings to keep connection alive."""
    try:
        while True:
            await asyncio.sleep(5)  # Send ping every 5 seconds
            try:
                await websocket.send(json.dumps({"ping": "ping"}))
                logger.debug("ping_sent", connection_id=connection_id)
            except websockets.exceptions.ConnectionClosed:
                break
    except asyncio.CancelledError:
        pass

async def connection_handler(websocket: websockets.WebSocketServerProtocol):
    """Handle websocket connections"""
    global connection_counter
    connection_id = connection_counter
    connection_counter += 1
    
    try:
        active_connections[connection_id] = websocket
        logger.info("client_connected", connection_id=connection_id)
        
        # Start ping task for this connection
        ping_task = asyncio.create_task(send_ping(websocket, connection_id))
        
        async for message in websocket:
            try:
                data = json.loads(message)
                await handle_message(websocket, connection_id, data)
            except json.JSONDecodeError:
                logger.error("json_decode_error", connection_id=connection_id)
                await websocket.send(json.dumps({
                    "success": False,
                    "error": "Invalid JSON format"
                }))
                
    except Exception as e:
        logger.error("websocket_error",
                    connection_id=connection_id,
                    error=str(e))
    finally:
        if connection_id in active_connections:
            del active_connections[connection_id]
            logger.info("client_disconnected", connection_id=connection_id)

async def main():
    logger.info("server_starting")
    async with websockets.serve(connection_handler, "localhost", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
