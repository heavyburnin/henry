
from tensor import Tensor

class AdamOptimizer:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [Tensor([0.0] * len(p.data)) for p in params]
        self.v = [Tensor([0.0] * len(p.data)) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i in range(len(self.params)):
            self.m[i] = self.m[i] * self.beta1 + grads[i] * (1 - self.beta1)
            self.v[i] = self.v[i] * self.beta2 + grads[i] * (1 - self.beta2)

            m_hat = self.m[i] * (1 / (1 - self.beta1 ** self.t))
            v_hat = self.v[i] * (1 / (1 - self.beta2 ** self.t))

            update = m_hat * (self.lr / ((v_hat + self.eps) ** 0.5))
            self.params[i] = self.params[i] + update
