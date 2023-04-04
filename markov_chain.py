import numpy as np


class MarkovChain:
    def __init__(self, matrix_size):
        self._states = np.zeros((matrix_size, matrix_size))
        self._matrix_size = matrix_size

    def __repr__(self):
        print(f"Chain with {self._matrix_size} states")
        print(self._states)

    def _check_chain_properties(self):
        return np.sum(self._states) == self._matrix_size

    def run(self, num_steps, initial_state, dec_val=1):
        if initial_state < 0 or initial_state >= len(f):
            return -1
        assert (
            self._check_chain_properties()
        ), "Chain does not have probabilities summing to 1"
        current_state = initial_state
        # Need this to accurately update current state
        state_range = np.arange(self._matrix_size)
        while num_steps > 0:
            current_state = np.random.choice(state_range, p=self._states[current_state])
            num_steps -= dec_val
