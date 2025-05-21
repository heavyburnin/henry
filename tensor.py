
class Tensor:
    def __init__(self, data):
        self.data = data

    def __add__(self, other):
        return Tensor([a + b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Tensor([a * scalar for a in self.data])

    def zero_(self):
        self.data = [0.0 for _ in self.data]

    def copy(self):
        return Tensor(self.data[:])

    def __repr__(self):
        return f"Tensor({self.data})"
