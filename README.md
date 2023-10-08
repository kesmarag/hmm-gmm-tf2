# HiddenMarkovModel Class Documentation

`HiddenMarkovModel` implements hidden Markov models with Gaussian mixtures as distributions on top of TensorFlow 2.0.

### Installation

```bash
pip install --upgrade git+https://gitlab.com/kesmarag/hmm-gmm-tf2
```

### Class Definition:

```python
HiddenMarkovModel(p0, tp, em_w, em_mu, em_var)
```

#### Arguments:

- **`p0`**: 1D numpy array
  - Determines the probability of the first hidden variable in the Markov chain for each hidden state. (e.g., `np.array([0.5, 0.25, 0.25])` for 3 hidden states)

- **`tp`**: 2D numpy array
  - Determines the transition probabilities for moving from one hidden state to another. (e.g., `np.array([[0.80, 0.15, 0.05], [0.20, 0.55, 0.25], [0.15, 0.15, 0.70]])` for 3 hidden states)

- **`em_w`**: 2D numpy array
  - Contains the weights of the Gaussian mixtures. Each line corresponds to a hidden state. (e.g., `np.array([[0.8, 0.2], [0.5, 0.5], [0.1, 0.9]])` for 3 hidden states, 2 Gaussian mixtures)

- **`em_mu`**: 3D numpy array
  - Determines the mean value vector for each component of the emission distributions. (e.g., `np.array([[[2.2, 1.3], [1.2, 0.2]], [[1.3, 5.0], [4.3, -2.3]], [[0.0, 1.2], [0.4, -2.0]]])` for 3 hidden states, 2 Gaussian mixtures)

- **`em_var`**: 3D numpy array
  - Determines the variance vector for each component of the emission distributions. (e.g., `np.array([[[2.2, 1.3], [1.2, 0.2]], [[1.3, 5.0], [4.3, -2.3]], [[0.0, 1.2], [0.4, -2.0]]])` for 3 hidden states, 2 Gaussian mixtures)

---

### `log_posterior(self, data)`

Log probability density function.

#### Arguments:

- **`data`**: 3D numpy array
  - The first dimension refers to each component of the batch.
  - The second dimension refers to each specific time interval.
  - The third dimension refers to the values of the observed data.

#### Returns:

- 1D numpy array with the values of the log-probability function with respect to the observations.

---

### `viterbi_algorithm(self, data)`

Performs the Viterbi algorithm for calculating the most probable hidden state path of some batch data.

#### Arguments:

- **`data`**: 3D numpy array
  - The first dimension refers to each component of the batch.
  - The second dimension refers to each specific time interval.
  - The third dimension refers to the values of the observed data.

#### Returns:

- 2D numpy array with the most probable hidden state paths.
  - The first dimension refers to each component of the batch.
  - The second dimension the order of the hidden states: (0, 1, ..., K-1), where K is the total number of hidden states.

---

### `fit(self, data, max_iter=100, min_var=0.01, verbose=False)`

Re-adapts the model parameters with respect to a batch of observations, using the Expectation-Maximization (E-M) algorithm.

#### Arguments:

- **`data`**: 3D numpy array
  - The first dimension refers to each component of the batch.
  - The second dimension refers to each specific time step.
  - The third dimension refers to the values of the observed data.

- **`max_iter`**: Positive integer number (default: 100)
  - The maximum number of iterations.

- **`min_var`**: Non-negative real value (default: 0.01)
  - The minimum acceptance variance. Used to prevent overfitting of the model.

- **`verbose`**: Boolean (default: False)
  - If True, prints detailed progress information during training.

#### Returns:

- 1D numpy array with the log-posterior probability densities for each training iteration.

---

### `generate(self, length, num_series=1, p=0.2)`

Generates a batch of time series.

#### Arguments:

- **`length`**: Positive integer
  - The length of each time series.

- **`num_series`**: Positive integer (default: 1)
  - The number of time series to generate.

- **`p`**: Real value between 0.0 and 1.0 (default: 0.2)
  - The importance sampling parameter.

#### Returns:

- 3D numpy array with the drawn time series.
- 2D numpy array with the corresponding hidden states.

---

### `kl_divergence(self, other, data)`

Estimates the value of the Kullback-Leibler divergence (KLD) between the model and another model with respect to some data.

#### Arguments:

- **`other`**: Another instance of the `HiddenMarkovModel` class for comparison.

- **`data`**: 3D numpy array representing observed data.

#### Returns:

- Estimated value of the Kullback-Leibler divergence (KLD) between the models with respect to the data.

---

### Example Usage:

```python
import numpy as np
from kesmarag.hmm import HiddenMarkovModel, new_left_to_right_hmm, store_hmm, restore_hmm, toy_example

dataset = toy_example()
model = new_left_to_right_hmm(states=3, mixtures=2, data=dataset)
model.fit(dataset, verbose=True)

# Store and restore the model
store_hmm(model, 'test_model.npz')
loaded_model = restore_hmm('test_model.npz')

# Generate data using the trained model
gen_data = model.generate(700, 10, 0.05)
```

This code demonstrates how to create, train, store, restore, and generate data using the `HiddenMarkovModel` class. The class provides functionality for working with hidden Markov models with Gaussian mixtures.

---
