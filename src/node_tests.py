from node import Node, PseudoNode_second_dim_divide, PseudoNode_second_dim_multiply
import unittest

class TestPseudoNode(unittest.TestCase):
    def test_multiply_forward_inference(self):
        node = PseudoNode_second_dim_multiply(factor=2)
        in_shape = (1, 4, 2)
        expected_out_shape = (1, 8, 2)
        self.assertEqual(node.forward_dimension_inference(in_shape), expected_out_shape)

    def test_multiply_backward_inference(self):
        node = PseudoNode_second_dim_multiply(factor=2)
        out_shape = (1, 8, 2)
        expected_in_shape = (1, 4, 2)
        self.assertEqual(node.backward_dimension_inference(out_shape), expected_in_shape)

    def test_divide_forward_inference(self):
        node = PseudoNode_second_dim_divide(factor=2)
        in_shape = (1, 8, 2)
        expected_out_shape = (1, 4, 2)
        self.assertEqual(node.forward_dimension_inference(in_shape), expected_out_shape)

    def test_divide_backward_inference(self):
        node = PseudoNode_second_dim_divide(factor=2)
        out_shape = (1, 4, 2)
        expected_in_shape = (1, 8, 2)
        self.assertEqual(node.backward_dimension_inference(out_shape), expected_in_shape)

if __name__ == '__main__':
    unittest.main()



