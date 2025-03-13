This project is going to cover a Visual/graph-compiler for all existing neuralnet architectures.
As well as graph representations that translate to PyTorch, JAX, Tensorflow model and training routine code. 
I will also develop another distributed training/inferencing framework based on this project. 
Please stay tuned! 

For Merge: we can have TensorConcatMerge, PositionwiseMerge(Like addition of Tensors), ContractionMerge(Like cross-product of Tensors) and OuterProductMerge(Like outer-product of Tensors). 

TensorConcatMerge takes: 1. the concat dimension(which dimension to concat the input tensors).  
We check that the input tensors share the same shape across all dimensions except on the concat dimensions, if not it throws an error. 

PositionwiseMerge takes:  1. a Position-wise op in the form of lambda, the default is addition x, y to x + 
y. 
We check if the input tensors share the same shape, if not it throws an error. 

ContractionMerge takes: 1. the Contraction dimension (which dimension of the input Tensors to perform the contraction dimension). 
2. A Position-wise op in the form of a lambda, the default is multiplication x,y to x*y. 
3. An aggregation op in the form of a lambda, the default is addition x,y to x + y. 
We check if the input tensors share the same shape, if not it throws an error. 

OuterProductMerge takes: 1. the OuterProduct dimension (which dimension of the input Tensors to perform the outerProduct) 
2. A Position-wise op in the form of a lambda, the default is multiplication x,y to x*y.  
We check if the input tensors share the same shape, if not it throws an error. 

For Branch: we can have TensorCopyBranch and TensorPartitionBranch. 

TensorCopyBranch just copy the Tensor to multiple downstream paths 

TensorPartitionBranch takes: 1. The dimension to partition the tensor on, 
2, the index to partition on that dimension. 
