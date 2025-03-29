from typing import Iterable, Iterator, Any, Tuple

class InvalidShapeError(Exception):
    """Exception raised when an invalid shape is specified."""
    def __init__(self, message: str = ""):
        default_message = "Invalid shape."
        super().__init__(f"{default_message} {message}")

class Shape:
    """
    Class representing the shape of a tensor.
    """
    def __init__(self, dimensions: Iterable[int]):
        """
        Initialize a Shape object with the given dimensions.
        
        Parameters:
        - dimensions: An iterable of integers representing the dimensions
        """
        self.dimensions = tuple(int(dim) for dim in dimensions)
        self._validate()
    
    def _validate(self):
        """
        Validate that all dimensions are valid.
        """
        for i, dim in enumerate(self.dimensions):
            if dim <= 0:
                raise InvalidShapeError(f"Invalid dimension at index {i}: {dim}. All dimensions must be positive.")
    
    def __eq__(self, other) -> bool:
        """
        Check if this shape is equal to another shape.
        """
        if isinstance(other, Shape):
            return self.dimensions == other.dimensions
        elif isinstance(other, tuple):
            return self.dimensions == other
        return False
    
    def __ne__(self, other) -> bool:
        """
        Check if this shape is not equal to another shape.
        """
        return not self.__eq__(other)
    
    def __len__(self) -> int:
        """
        Get the number of dimensions.
        """
        return len(self.dimensions)
    
    def __getitem__(self, index) -> int:
        """
        Get a dimension by index.
        """
        return self.dimensions[index]
    
    def __iter__(self) -> Iterator[int]:
        """
        Iterate over the dimensions.
        """
        return iter(self.dimensions)
    
    def __repr__(self) -> str:
        """
        Get a string representation of the shape.
        """
        return f"Shape{self.dimensions}"
    
    def __str__(self) -> str:
        """
        Get a string representation of the shape.
        """
        return str(self.dimensions)
    
    def total_elements(self) -> int:
        """
        Calculate the total number of elements in a tensor with this shape.
        """
        if not self.dimensions:
            return 0
        total = 1
        for dim in self.dimensions:
            total *= dim
        return total 