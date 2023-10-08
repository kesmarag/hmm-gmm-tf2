import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from kesmarag.hmm.dataset import DataSet
# import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time
# disable the tensorflow's warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(6)

def dataset_tf(dataset):
  return dataset

def new_left_to_right_hmm(states, mixtures, data):
  '''### Function for creating a new Left-to-Right HMM
  '''
  N = data.shape[1]
  Ns = N // states
  D = data.shape[-1]
  mu = []
  for k in range(states):
    kmeans_batch = np.squeeze(data)[(k*Ns):(k*Ns+Ns)]
    kmeans = KMeans(n_clusters=mixtures)
    kmeans.fit(kmeans_batch)
    mu.append(kmeans.cluster_centers_)
  mu = np.array(mu)
  var = 3.0 * np.ones((states, mixtures, D))
  w = np.ones((states, mixtures))/mixtures
  p0 = np.zeros(states)
  p0[0] = 1.0
  tp = np.zeros((states, states))
  tpkk = 1.0 - 1.0/Ns
  for k in range(states-1):
    tp[k,k]=tpkk
    tp[k,k+1]=1.0-tpkk
  tp[-1,-1]=1.0
  model = HiddenMarkovModel(p0, tp, w, mu, var)
  return model

def store_hmm(model, filename):
  '''Saves a HMM to the disk.
  '''
  np.savez(filename,
           p0 = model._p0,
           tp = model._tp,
           w = model._em_w,
           mu = model._em_mu,
           var = model._em_var)

def restore_hmm(filename):
  '''Loads a saved HMM from a filename.
  '''
  z = np.load(filename)
  p0 = z['p0']
  tp = z['tp']
  w = z['w']
  mu = z['mu']
  var = z['var']
  return HiddenMarkovModel(p0, tp, w, mu, var)

def toy_example():
  '''A dataset example
  '''
  N3 = [200, 300, 200]
  s = 0
  mn = np.array([[[0.5, 1.], [2., 1.]],
                 [[2., 5.], [4., 5.]],
                 [[4., 1.], [6., 5.]]])
  var = np.array([[[1., 1.], [1.2, 1.]],
                  [[.8, 1.], [.8, 1.]],
                  [[1., 1.], [.8, 1.2]]])
  samples = np.zeros((sum(N3), 2))
  for i in range(3):
    em = tfp.distributions.MixtureSameFamily(
       mixture_distribution=tfp.distributions.Categorical(
           probs=np.array([0.6, 0.4])),
       components_distribution=tfp.distributions.MultivariateNormalDiag(
           loc=mn[i],
           scale_diag=var[i]))
    st = s + N3[i]
    samples[s:st,:] = em.sample(N3[i]).numpy()
    s += N3[i]
  return np.expand_dims(samples, 0)

class HiddenMarkovModel(object):
  """**HiddenMarkovModel implements hidden Markov models with Gaussian mixtures as**
     **distributions on top of TensorFlow 2.0**
  """
  def __init__(self, p0, tp, em_w, em_mu, em_var):
    """
    Args:
      p0: 1D numpy array
        Determines the probability of the first hidden variable
        in the Markov chain for each hidden state.
        e.g. np.array([0.5, 0.25, 0.25]) (3 hidden states)
      tp: 2D numpy array
        Determines the transition probabilities for moving from one hidden state to each
        other. The (i,j) element of the matrix denotes the probability of
        transiting from i-th state to the j-th state.
        e.g. np.array([[0.80, 0.15, 0.05],
                       [0.20, 0.55, 0.25],
                       [0.15, 0.15, 0.70]])
        (3 hidden states)
      em_w: 2D numpy array
        Contains the weights of the Gaussian mixtures.
        Each line correspond to a hidden state.
        e.g. np.array([[0.8, 0.2],
                       [0.5, 0.5],
                       [0.1, 0.9]])
        (3 hidden states, 2 Gaussian mixtures)
      em_mu: 3D numpy array
        Determines the mean value vector for each component
        of the emission distributions.
        The first dimension refers to the hidden states whereas the
        second one refer to the mixtures.
        e.g. np.array([[[2.2, 1.3], [1.2, 0.2]],    1st hidden state
                       [[1.3, 5.0], [4.3, -2.3]],   2nd hidden state
                       [[0.0, 1.2], [0.4, -2.0]]])  3rd hidden state
        (3 hidden states, 2 Gaussian mixtures)
      em_var: 3D numpy array
        Determines the variance vector for each component of the
        emission distributions.
        e.g. np.array([[[2.2, 1.3], [1.2, 0.2]],    1st hidden state
                       [[1.3, 5.0], [4.3, -2.3]],   2nd hidden state
                       [[0.0, 1.2], [0.4, -2.0]]])  3rd hidden state
        (3 hidden states, 2 Gaussian mixtures)
    """
    self._p0 = p0
    self._tp = tp
    self._em_w = em_w
    self._em_mu = em_mu
    self._em_var = em_var
    self._k = tp.shape[0] # number of hidden states

  @property
  def p0(self):
    return np.squeeze(self._p0)

  @property
  def tp(self):
    return self._tp

  @property
  def w(self):
    return np.squeeze(self._em_w)

  @property
  def mu(self):
    return self._em_mu

  @property
  def var(self):
    return self._em_var

  def __str__(self):
    s = '### [kesmarag.hmm.HiddenMarkovModel] ###'
    s += '\n\n=== Prior probabilities ================\n\n'
    s += str(self.p0)
    s += '\n\n=== Transition probabilities ===========\n\n'
    s += str(self.tp)
    s += '\n\n=== Emission distributions =============\n \n'
    for i in range(self.w.shape[0]):
      if i != 0:
        s += '\n'
      s += '*** Hidden state #' + str(i+1) + ' *** \n \n'
      for j in range(self.w.shape[1]):
        s += '--- Mixture #' + str(j+1) + ' --- \n'
        s += 'weight : ' + str(self.w[i, j])
        s += '\n'
        s += 'mean_values : ' + str(self.mu[i,j,:])
        s += '\n'
        s += 'variances : ' + str(self.var[i,j,:])
        s += '\n'
        if j != self.w.shape[1] - 1:
          s += '\n'
    return s

  def log_posterior(self, data):
    ''' Log probability density function.

        Args:
          data: 3D numpy array
            The first dimension refers to each component of the batch.
            The second dimension refers to each specific time interval.
            The third dimension refers to the values of the observed data.

        Returns:
          1D numpy array with the values of the log-probability function with respect to the observations.
    '''
    _, _, posterior = self._forward(data,
                                    self._p0,
                                    self._tp,
                                    self._em_w,
                                    self._em_mu,
                                    self._em_var)
    return np.squeeze(posterior.numpy())

  def viterbi_algorithm(self, data):
    ''' Performs the viterbi algorithm for calculating the most probable
        hidden state path of some batch data.

        Args:
          data: 3D numpy array
            The first dimension refers to each component of the batch.
            The second dimension refers to each specific time interval.
            The third dimension refers to the values of the observed data.

        Returns:
          2D numpy array with the most probable hidden state paths.
            The first dimension refers to each component of the batch.
            The second dimension the order of the hidden states.
            (0, 1, ..., K-1), where K is the total number of hidden states.
    '''
    w_tf, am_tf = self._viterbi(data, self._p0, self._tp, self._em_w, self._em_mu, self._em_var)
    w = np.array(w_tf)[:, -1, :]
    am = np.array(am_tf)
    argmax_w = np.argmax(w, axis=1)
    psi = np.concatenate((am[range(len(w)), 1::, argmax_w], np.expand_dims(argmax_w, 1)), -1)
    dec = []
    for i, p in enumerate(am):
      dec_p = [argmax_w[i]]
      c = p[::-1, :]
      l = argmax_w[i]
      for j in range(len(c) - 2):
        dec_p.insert(0, c[j][l])
        l = c[j][l]
      dec_p.insert(0, l)
      dec.append(dec_p)
    return np.squeeze(np.array(dec, dtype='int16'))

  def _forward_step(self, n, alpha, c, data, _tp, em_w, em_mu, em_var):
    alpha_tp = tf.matmul(alpha[n-1], _tp)
    em = tfp.distributions.MixtureSameFamily(
     mixture_distribution=tfp.distributions.Categorical(
         probs=em_w),
     components_distribution=tfp.distributions.MultivariateNormalDiag(
         loc=em_mu,
         scale_diag=em_var))
    ds = tf.expand_dims(dataset_tf(data), -2)
    a_n_tmp = tf.multiply(em.prob(ds[:, n, :]), alpha_tp)
    c_n_tmp = tf.expand_dims(tf.reduce_sum(input_tensor=a_n_tmp, axis=-1), -1)
    return [n + 1, tf.concat([alpha, tf.expand_dims(a_n_tmp / c_n_tmp, 0)], 0),
            tf.concat([c, tf.expand_dims(c_n_tmp, 0)], 0),
            data, _tp, em_w, em_mu, em_var]

  @tf.function
  def _forward(self, data, _p0, _tp, em_w, em_mu, em_var):
    n = tf.shape(input=data)[1]
    em = tfp.distributions.MixtureSameFamily(
     mixture_distribution=tfp.distributions.Categorical(
         probs=em_w),
     components_distribution=tfp.distributions.MultivariateNormalDiag(
         loc=em_mu,
         scale_diag=em_var))
    ds = tf.expand_dims(dataset_tf(data), -2)
    a_0_tmp = tf.expand_dims(
      tf.multiply(em.prob(ds[:, 0, :]), tf.squeeze(_p0)), 0)
    c_0 = tf.expand_dims(tf.reduce_sum(input_tensor=a_0_tmp, axis=-1), -1)
    alpha_0 = a_0_tmp / c_0
    i0 = tf.constant(1)
    condition_forward = lambda i, alpha, c, data, tp, em_w, em_mu, em_var: tf.less(i, n)
    _, alpha, c, _, _, _, _, _ = \
        tf.while_loop(
          cond=condition_forward,
          body=self._forward_step,
          loop_vars=[i0, alpha_0, c_0, data, _tp, em_w, em_mu, em_var],
          shape_invariants=[
            i0.get_shape(),
            tf.TensorShape(
              [None, None, self._k]),
            tf.TensorShape([None, None, 1]),
            data.shape,
            _tp.shape,
            em_w.shape,
            em_mu.shape,
            em_var.shape])
    posterior = tf.reduce_sum(input_tensor=tf.math.log(c), axis=0,
                                    name='log_post')
    return alpha, c, posterior

  def _backward_step(self, n, betta, b_p, data, _tp, em_w, em_mu, em_var):
    ds = tf.expand_dims(dataset_tf(data), -2)
    em = tfp.distributions.MixtureSameFamily(
     mixture_distribution=tfp.distributions.Categorical(
         probs=em_w),
     components_distribution=tfp.distributions.MultivariateNormalDiag(
         loc=em_mu,
         scale_diag=em_var))
    b_p_tmp = tf.multiply(betta[0],
                          tf.squeeze(em.prob(ds[:, -n, :])))
    b_n_tmp = tf.matmul(b_p_tmp, _tp, transpose_b=True) / self._c[-n]
    return [n + 1, tf.concat([tf.expand_dims(b_n_tmp, 0), betta], 0),
            tf.concat([tf.expand_dims(b_p_tmp, 0), b_p], 0),
            data, _tp, em_w, em_mu, em_var]

  @tf.function
  def _backward(self, data, _p0, _tp, em_w, em_mu, em_var):
    ds = tf.expand_dims(dataset_tf(data), -2)
    n = tf.shape(input=ds)[1]
    shape = tf.shape(input=ds)[0]
    dims = tf.stack([shape, self._k])
    b_tmp_ = tf.fill(dims, 1.0)
    betta_0 = tf.expand_dims(tf.ones_like(b_tmp_, dtype=tf.float64), 0)
    b_p_0 = tf.expand_dims(tf.ones_like(b_tmp_, dtype=tf.float64), 0)
    i0 = tf.constant(1)
    condition_backward = lambda i, betta, b_p, data, _tp, em_w, em_mu, em_var: tf.less(i, n)
    _, betta, b_p_tmp, _, _, _, _, _= \
        tf.while_loop(
          cond=condition_backward,
          body=self._backward_step,
          loop_vars=[i0, betta_0, b_p_0, data, _tp, em_w, em_mu, em_var],
          shape_invariants=[
            i0.get_shape(),
            tf.TensorShape([None, None, self._k]),
            tf.TensorShape([None, None, self._k]),
            data.shape,
            _tp.shape,
            em_w.shape,
            em_mu.shape,
            em_var.shape])
    b_p = b_p_tmp[:-1, :, :]
    return betta, b_p

  def _viterbi_step(self, n, w, am, data, _tp, em_w, em_mu, em_var):
    ds = tf.expand_dims(dataset_tf(data), -2)
    em = tfp.distributions.MixtureSameFamily(
     mixture_distribution=tfp.distributions.Categorical(
         probs=em_w),
     components_distribution=tfp.distributions.MultivariateNormalDiag(
         loc=em_mu,
         scale_diag=em_var))
    w_tmp = tf.expand_dims(tf.math.log(em.prob(ds)[:, n]) + tf.reduce_max(
          input_tensor=tf.expand_dims(w[:, n - 1], -1) +
      tf.expand_dims((tf.math.log(_tp)), 0) , axis=-2), 1)
    am_tmp = tf.expand_dims(tf.argmax(input=tf.expand_dims(w[:, n - 1], -1) +
                                      tf.expand_dims((tf.math.log(_tp)), 0) , axis=-2), 1)
    return [n + 1, tf.concat([w, w_tmp], 1), tf.concat([am, am_tmp], 1),
            data, _tp, em_w, em_mu, em_var]

  @tf.function
  def _viterbi(self, data, _p0, _tp, em_w, em_mu, em_var):
    ds = tf.expand_dims(dataset_tf(data), -2)
    m = tf.shape(input=ds)[0]
    n = tf.shape(input=ds)[1]
    em = tfp.distributions.MixtureSameFamily(
     mixture_distribution=tfp.distributions.Categorical(
         probs=em_w),
     components_distribution=tfp.distributions.MultivariateNormalDiag(
         loc=em_mu,
         scale_diag=em_var))
    w1 = tf.expand_dims(
      tf.math.log(_p0) + tf.math.log(em.prob(ds)[:, 0]), 1)
    am1 = tf.zeros_like(w1, dtype='int64')
    i0 = tf.constant(1)
    condition_viterbi = lambda i, w, am, data, _tp, em_w, em_mu, em_var: tf.less(i, n)
    _, w, am, _, _, _, _, _ = tf.while_loop(
      cond=condition_viterbi, body=self._viterbi_step, loop_vars=[i0, w1, am1, data, _tp, em_w, em_mu, em_var],
      shape_invariants=[i0.get_shape(),
                        tf.TensorShape([None, None, self._k]),
                        tf.TensorShape([None, None, self._k]),
                        data.shape,
                        _tp.shape,
                        em_w.shape,
                        em_mu.shape,
                        em_var.shape])
    return w, am

  def _xi_calc(self, n, xi, _tp):
    a_b_p = tf.matmul(
      tf.expand_dims(self._alpha[n - 1] / self._c[n], -1),
      tf.expand_dims(self._b_p[n - 1], -1), transpose_b=True)
    xi_n_tmp = tf.multiply(a_b_p, _tp)
    return [n + 1, tf.concat([xi, tf.expand_dims(xi_n_tmp, 0)], 0), _tp]

  @tf.function
  def _expectation(self, data, _p0, _tp, em_w, em_mu, em_var):
    self._alpha, self._c, self._posterior = self._forward(data, _p0, _tp, em_w, em_mu, em_var)
    self._betta, self._b_p = self._backward(data, _p0, _tp, em_w, em_mu, em_var)
    # gamma shape : (N, I, states)
    ds = tf.expand_dims(dataset_tf(data), -2)
    gamma = tf.multiply(self._alpha, self._betta, name='gamma')
    n = tf.shape(input=ds)[1]
    shape = tf.shape(input=ds)[0]
    dims = tf.stack([shape, self._k, self._k])
    xi_tmp_ = tf.fill(dims, 1.0)
    xi_0 = tf.expand_dims(tf.ones_like(xi_tmp_, dtype=tf.float64), 0)
    i0 = tf.constant(1)
    condition_xi = lambda i, xi, _tp: tf.less(i, n)
    _, xi_tmp, _ = tf.while_loop(
      cond=condition_xi, body=self._xi_calc, loop_vars=[i0, xi_0, _tp],
      shape_invariants=[i0.get_shape(), tf.TensorShape(
        [None, None, self._k, self._k]), _tp.shape])
    xi = xi_tmp[1:, :, :]
    return gamma, xi

  def fit(self, data, max_iter=100, min_var=0.01, verbose=False):
    ''' This method re-adapts the model parameters with respect to a batch of
        observations, using the Expectation-Maximization (E-M) algorithm.

        Args:
          data: 3D numpy array
            The first dimension refers to each component of the batch.
            The second dimension refers to each specific time step.
            The third dimension refers to the values of the observed data.
          max_iter: positive integer number
            The maximum number of iterations.
          min_var: non-negative real value
            The minimum acceptance variance. We use this restriction
            in order to prevent overfitting of the model.

        Returns:
          1D numpy array with the log-posterior probability densities for each training iteration.
    '''
    posts = []
    _, _, posterior = self._forward(data,self._p0, self._tp, self._em_w, self._em_mu,
                                      self._em_var)
    posts.append(np.mean(posterior.numpy()))
    for i in range(max_iter):
      if verbose:
        print('epoch:', str(i).rjust(3), ', ln[p(X|λ)] =', posts[-1])
      p0_new, tp_new, w_new, mu_new, cov_new = self._maximization(data, self._p0,
                                                                  self._tp, self._em_w,
                                                                  self._em_mu,
                                                                  self._em_var,
                                                                  min_var)
      self._p0 = p0_new.numpy()
      self._tp = tp_new.numpy()
      self._em_w = w_new.numpy()
      self._em_mu = mu_new.numpy()
      self._em_var = cov_new.numpy()

      _, _, posterior = self._forward(data, self._p0, self._tp, self._em_w, self._em_mu,
                                      self._em_var)
      posts.append(np.mean(posterior.numpy()))
      if np.abs(posts[-1]-posts[-2])/(np.abs(posts[-2])/data.shape[1]) < 0.01:
        if verbose:
          print('epoch:', str(i+1).rjust(3), ', ln[p(X|λ)] =', posts[-1])
        break
      elif i==max_iter-1 and verbose:
        print('epoch:', str(i+1).rjust(3), ', ln[p(X|λ)] =', posts[-1])
    return posts

  @tf.function
  def _maximization(self, data, _p0, _tp, em_w, em_mu, em_var, min_var):
    gamma, xi = self._expectation(data, _p0, _tp, em_w, em_mu, em_var)
    gamma_mv = tf.reduce_mean(input_tensor=gamma, axis=1, name='gamma_mv')
    xi_mv = tf.reduce_mean(input_tensor=xi, axis=1, name='xi_mv')
    # update the initial state probabilities
    p0_new = tf.transpose(a=tf.expand_dims(gamma_mv[0], -1))
    # update the transition matrix
    # first calculate sum_n=2^{N} xi_mean[n-1,k , n,l]
    sum_xi_mean = tf.squeeze(tf.reduce_sum(input_tensor=xi_mv, axis=0))
    tp_new = sum_xi_mean / (tf.reduce_sum(input_tensor=sum_xi_mean,
                                                   axis=1,
                                                   keepdims=True))
    em = tfp.distributions.MixtureSameFamily(
     mixture_distribution=tfp.distributions.Categorical(
         probs=em_w),
     components_distribution=tfp.distributions.MultivariateNormalDiag(
         loc=em_mu,
         scale_diag=em_var))
    dss = tf.expand_dims(dataset_tf(data), -2)
    dsss = tf.expand_dims(dss, -2)
    tt_tmp = em.mixture_distribution.probs * em.components_distribution.prob(dsss) / tf.expand_dims(em.prob(dss),-1)
    tt = tf.where(tf.math.is_nan(tt_tmp), tf.zeros_like(tt_tmp), tt_tmp)
    ttt = tf.transpose(a=tt, perm=[1, 0, 3, 2])
    tttt = ttt * tf.expand_dims(gamma, -2)
    gamma_mv_t= tf.reduce_mean(input_tensor=tttt, axis=1)
    # update mixture models
    gg = tf.reduce_mean(input_tensor=gamma_mv_t, axis=0)
    w_new = tf.transpose(gg / tf.reduce_sum(input_tensor=gg, axis=0))
    x_t = tf.transpose(a=dataset_tf(data), perm=[1, 0, 2], name='x_transpose')

    x_t = tf.expand_dims(x_t, -2)
    gamma_x = tf.matmul(tf.expand_dims(tttt, -1),
                        tf.expand_dims(x_t, -1), transpose_b=True)
    sum_gamma_x = tf.transpose(tf.reduce_sum(input_tensor=gamma_x, axis=[0, 1]), perm=[1,
                                                                                       0,
                                                                                       2])

    mu_tmp_t = tf.transpose(a=sum_gamma_x) / tf.reduce_sum(
      input_tensor=tttt,
      axis=[0, 1])

    mu_new = tf.transpose(a=mu_tmp_t)

    x_expanded = tf.expand_dims(dataset_tf(data), -2)
    x_m_mu = tf.subtract(tf.expand_dims(x_expanded, -2), em_mu)
    x_m_mu_2 = tf.matmul(tf.expand_dims(x_m_mu, -1),
                         tf.expand_dims(x_m_mu, -1),transpose_b=True)
    gamma_r = tf.transpose(a=tttt, perm=[1, 0, 3, 2])

    gamma_x_m_mu_2 = tf.multiply(
                        x_m_mu_2,
                        tf.expand_dims(tf.expand_dims(gamma_r, -1), -1)
    )
    cov_full_new = tf.reduce_sum(
      input_tensor=gamma_x_m_mu_2,
      axis=[0, 1]) / tf.expand_dims(
      tf.expand_dims(
        tf.reduce_sum(
          input_tensor=gamma_r,
          axis=[0, 1]), -1), -1)
    cov_new = tf.maximum(tf.sqrt(tf.linalg.diag_part(cov_full_new)), min_var)
    return p0_new, tp_new, w_new, mu_new, cov_new

  @tf.function
  def _generate_single_step(self, num_series, prev_state, _tp, em_w, em_mu, em_var):
    tp = tfp.distributions.Categorical(probs=_tp)
    full_next_state = tp.sample((num_series,1))
    series_order = tf.expand_dims(tf.range(num_series), -1)
    prev_state_with_order = tf.concat([series_order, tf.expand_dims(prev_state, -1)], 1)
    next_state = tf.gather_nd(full_next_state[:,0], prev_state_with_order)
    next_state_with_order = tf.concat([series_order, tf.expand_dims(next_state, -1)], 1)
    em = tfp.distributions.MixtureSameFamily(
      mixture_distribution=tfp.distributions.Categorical(
        probs=em_w),
      components_distribution=tfp.distributions.MultivariateNormalDiag(
        loc=em_mu,
        scale_diag=em_var))
    full_samples = em.sample(num_series)
    obs = tf.gather_nd(full_samples, tf.expand_dims(next_state_with_order, 0))
    return next_state, obs

  @tf.function
  def _generate_first_step(self, num_series, _p0, em_w, em_mu, em_var):
    p0 = tfp.distributions.Categorical(probs=_p0)
    init_state = tf.squeeze(p0.sample((num_series, 1)))
    if num_series == 1:
      init_state = tf.expand_dims(init_state, 0)
    series_order = tf.expand_dims(tf.range(num_series), -1)
    init_state_with_order = tf.concat([series_order, tf.expand_dims(init_state, -1)], 1)
    em = tfp.distributions.MixtureSameFamily(
      mixture_distribution=tfp.distributions.Categorical(
        probs=em_w),
      components_distribution=tfp.distributions.MultivariateNormalDiag(
        loc=em_mu,
        scale_diag=em_var))
    full_samples = em.sample(num_series)
    obs = tf.gather_nd(full_samples, tf.expand_dims(init_state_with_order, 0))
    return init_state, obs


  def generate(self, length, num_series=1, p=0.2):
    '''Generates a batch of time series.

    Args:
      length: positive integer
        The length of each time series.
      num_series: positive integer (default 1)
        The number of the time series.
      p: real value between 0.0 and 1.0 (default 0.2)
        The importance sampling parameter.
        At each iteration:
      k[A] Draw X and calculate p(X)
          if p(X) > p(X_{q-1}) then
            accept X as X_q
          else
            draw r from [0,1] using the uniform distribution.
            if r > p then
              accept the best of the rejected ones.
            else
              go to [A]

    Returns:
      3D numpy array with the drawn time series.
      2D numpy array with the corresponding hidden states.

    '''
    samples = np.zeros((length, num_series, self._em_mu.shape[-1]))
    states = np.zeros((length, num_series))
    # initial_state = np.zeros((num_series), 'int32')
    n = 0
    prev = -1e10
    while n < num_series:
      best_post_n = -1e11
      cur_state, obs = self._generate_first_step(1,
                                                 self._p0,
                                                 self._em_w,
                                                 self._em_mu,
                                                 self._em_var)
      samples[0] = obs.numpy()
      for l in range(length - 1):
        state, obs = self._generate_single_step(1,
                                                cur_state,
                                                self._tp,
                                                self._em_w,
                                                self._em_mu,
                                                self._em_var)
        samples[l + 1, n, :] = obs.numpy()
        cur_state = state.numpy()
        states[l + 1, n] = cur_state
      post = self.log_posterior(np.expand_dims(samples[:, n, :], 0))
      if post > best_post_n:
        best_post_n = post
        best_samples_n = obs.numpy()
      if best_post_n > prev or np.random.rand() < p:
        print(n , best_post_n)
        samples[l + 1, n, :] = best_samples_n
        n += 1
        prev = best_post_n
      else:
        pass
    samples = np.transpose(samples, [1, 0, 2])
    states = np.transpose(states, [1, 0])
    return samples, states

  def kl_divergence(self, other, data):
    ''' Estimates the value of the Kullback-Leibler divergence (KLD)
        between the model and another model with respect to some data.
    '''
    return np.mean(self.log_posterior(data) - other.log_posterior(data)) / data.shape[1]


if __name__ == '__main__':
  toy_data = toy_example()

  model = new_left_to_right_hmm(3, 2, toy_data)
  other = new_left_to_right_hmm(3, 2, toy_data)

  model.fit(toy_data, max_iter=50, verbose=True, min_var = 0.001)

  samples1, states1= model.generate(700, 1)
  print(model.kl_divergence(other, samples1))
  import matplotlib.pyplot as plt
  plt.plot(samples1[0])
  plt.savefig('test.jpg')
