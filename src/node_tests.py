from node import Node, Reshape, Tensor
import unittest
import torch
import tensorflow as tf
import numpy as np


#TODO Some notes to front-end: 
#the bubble sould only look green if a node has in_shape, out_shape, input_node and output_node all defined
#call .completed() to check


class TestNodeOperations(unittest.TestCase):
    def test_successful_reshape(self): 
        # Create a tensor with shape (3, 8, 3)
        data = torch.randn(3, 8, 3)
        tensor_node = Tensor(data)
        print(tensor_node.in_shape)
        print(tensor_node.out_shape)
        # Apply reshape (3, 4, -1)
        reshape_node1 = Reshape((3, 4, -1))
        reshape_node1.set_input_node(tensor_node)
        self.assertEqual(reshape_node1.out_shape, (3, 4, 6))

        # Apply reshape (3, -1, 2)
        reshape_node2 = Reshape((3, -1, 2))
        reshape_node2.set_input_node(reshape_node1)
        self.assertEqual(reshape_node2.out_shape, (3, 12, 2))

    def test_reshape_failure(self):
        # Create a tensor with shape (3, 8, 3)
        data = torch.randn(3, 8, 3)
        tensor_node = Tensor(data)

        # Attempt invalid reshape (3, 5, -1)
        reshape_node = Reshape((3, 5, -1))
        with self.assertRaises(ValueError) as context:
            reshape_node.set_input_node(tensor_node)
        self.assertIn("Cannot infer dimension", str(context.exception))

    def test_dimension_mismatch(self):
        # Create a tensor with shape (3, 8, 3)
        data = torch.randn(3, 8, 3)
        tensor_node = Tensor(data)

        # Apply valid reshape (3, 4, 6)
        reshape_node = Reshape((3, 4, 6))
        reshape_node.set_input_node(tensor_node)

        # Attempt to connect a tensor with a different shape
        different_tensor = Tensor(torch.randn(3, 10, 2))
        with self.assertRaises(ValueError) as context:
            reshape_node.set_output_node(different_tensor)
        self.assertIn("out_shape mismatch", str(context.exception))

if __name__ == '__main__':
    unittest.main()



