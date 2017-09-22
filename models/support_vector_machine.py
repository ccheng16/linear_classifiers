"""
Implements support vector machine.
"""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class SupportVectorMachine(LinearModel):
    def backward(self, f, y):
        """Performs the backward operation.

        By backward operation, it means to compute the gradient of the loss
        w.r.t w.

        Hint: You may need to use self.x, and you made need to change the
        forward operation.

        x(numpy.ndarray): Dimension of (N, ndims + 1), N is the number of examples.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
            y(numpy.ndarray): Ground truth label, dimension (N,).
        Returns:
            (numpy.ndarray): Gradient of L w.r.t to self.w,
              dimension (ndims+1,).
        """
        c = 0.5
        return 2 * c * self.w - np.mean(self.x.T * y * (y * f <= 1), axis=1)

    def loss(self, f, y):
        """The average loss across batch examples.
        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
            y(numpy.ndarray): Ground truth label, dimension (N,).
        Returns:
            (float): average hinge loss.
        """
        c = 0.5
        return c * np.sum(self.w ** 2) + np.mean((1 - y * f) * (y * f < 1))

    def predict(self, f):
        """
        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
        Returns:
            (numpy.ndarray): Hard predictions from the score, f,
              dimension (N,).
        """
        return 1 * (f > 0) - 1 * (f < 0)
