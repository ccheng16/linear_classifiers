"""
Implements logistic regression.
"""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class LogisticRegression(LinearModel):
    """
    """
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
        denom = 1 + np.exp(f * y)
        num = -self.x.T * y
        return np.mean(num / denom, axis=1)

    def loss(self, f, y):
        """The average loss across batch examples.
        Args:
        f(numpy.ndarray): Output of forward operation, dimension (N,).
        y(numpy.ndarray): Ground truth label, dimension (N,).
        Returns:
        (float): average log loss.
        """
        return np.mean(np.log(1 + np.exp(-f * y)))

    def predict(self, f):
        """
        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
        Returns:
            (numpy.ndarray): Hard predictions from the score, f,
              dimension (N,).
        """
        sigf = 1 / (1 + np.exp(-f))
        return 1 * (sigf > 0.5) - 1 * (sigf < 0.5)