import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K

class SinkhornDistance(Loss):
    def __init__(self, eps, max_iter, p=2, reduction='none'):
        super(SinkhornDistance, self).__init__(reduction=reduction)
        self.eps = eps
        self.max_iter = max_iter
        self.p = p

    def call(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y) # Wasserstein cost function
        x_points = tf.shape(x)[-2].numpy()
        y_points = tf.shape(y)[-2].numpy()
        if tf.rank(x) == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = tf.fill((batch_size, x_points), 1.0 / x_points)
        nu = tf.fill((batch_size, y_points), 1.0 / y_points)
        mu = tf.squeeze(mu)
        nu = tf.squeeze(nu)

        u = tf.zeros_like(mu, dtype=tf.float64)
        v = tf.zeros_like(nu, dtype=tf.float64)

        # Sinkhorn iterations
        # The algorithm can stop due to maximum iterations or due to threshold
        for i in range(self.max_iter):
            u1 = u
            u = self.eps * (tf.math.log(mu + 1e-8) - tf.reduce_logsumexp(self.M(C, u, v), axis=-1)) + u
            v = self.eps * (tf.math.log(nu + 1e-8) - tf.reduce_logsumexp(self.M(C, u, v), axis=-2)) + v
            err = K.sum(K.abs(u - u1), axis=-1)

            # Stopping criterion due to threshold
            if K.mean(err) < 1e-1:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = tf.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = K.sum(pi * C, axis=(-2, -1))

        if self.reduction == 'mean':
            cost = K.mean(cost)
        elif self.reduction == 'sum':
            cost = K.sum(cost)

        return cost, pi, C

    def M(self, C, u, v):
        return (-C + tf.expand_dims(u, axis=-1) + tf.expand_dims(v, axis=-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        x_col = tf.expand_dims(x, axis=-2)
        y_lin = tf.expand_dims(y, axis=-3)
        C = K.sum(K.abs(x_col - y_lin) ** p, axis=-1)
        C = tf.cast(C, dtype=tf.float64)
        return C

class AdversarialSinkhornDivergence(Loss):
    def __init__(self, epsilon=0.1, max_iter=50, reduction='none'):
        super(AdversarialSinkhornDivergence, self).__init__(reduction=reduction)
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.crossentropy = tf.keras.losses.CategoricalCrossentropy()

    def call(self, outputs_clean, outputs_adv, target):
        loss_adv = self.crossentropy(target, outputs_adv)
        loss_adv = tf.cast(loss_adv, dtype=tf.float64)

        sink = SinkhornDistance(eps=self.epsilon, max_iter=self.max_iter)

        dist, _, _ = sink.call(outputs_clean, outputs_adv)
        dist_adv, _, _ = sink.call(outputs_adv, outputs_adv)
        dist_clean, _, _ = sink.call(outputs_clean, outputs_clean)
        dist = dist - 0.5 * (dist_adv + dist_clean)

        loss_sink = dist

        return loss_adv, loss_sink
