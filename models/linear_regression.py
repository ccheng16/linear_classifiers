"""
Implements lienar regression.
"""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class LinearRegression(LinearModel):
    """Implements a linear regression mode model"""

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
        return np.mean(self.x.T * (f - y), axis=1)

    def loss(self, f, y):
        """The average loss across batch examples.
        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
            y(numpy.ndarray): Ground truth label, dimension (N,).
        Returns:
            (float): average square loss.
        """
        return np.mean((f - y) ** 2)

    def predict(self, f):
        """Converts score into predictions in {-1, 1}.
        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
        Returns:
            (numpy.ndarray): Hard predictions from the score, f,
              dimension (N,).
        """
        y_predict = 1 * (f > 0) - 1 * (f < 0)
        return y_predict
