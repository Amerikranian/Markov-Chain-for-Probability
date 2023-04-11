import numpy as np


class MarkovChain:
    def __init__(self, matrix_size):
        self._states = np.zeros((matrix_size, matrix_size))
        self._matrix_size = matrix_size
        self._state_labels = {}

    def __repr__(self):
        chain_str = f"Chain with {self._matrix_size} states\n"
        for state_num, state_name in self._state_labels.items():
            chain_str += f"{state_name}: {self._states[state_num]}\n"
        return chain_str

    def add(self, state, state_name, values):
        assert (
            values.shape[0] == self._matrix_size
        ), "Not enough probabilities provided for all states in the simulation"
        assert np.sum(values) == 1, "Probabilities do not sum to one"
        self._state_labels[state] = state_name
        self._states[state] = values

    def get_label_for_state(self, state):
        return self._state_labels.get(state)

    def run(self, num_steps, initial_state, dec_val=1, sim_callback=None):
        if initial_state < 0 or initial_state >= len(self._states):
            return -1
        current_state = initial_state
        # Need this to accurately update current state
        state_range = np.arange(self._matrix_size)
        while num_steps > 0:
            current_state = np.random.choice(state_range, p=self._states[current_state])
            num_steps -= dec_val
            if sim_callback is not None:
                temp = sim_callback(self, current_state, num_steps, dec_val)
                if temp != -1:
                    # Something happened in the chain that we need to fix
                    num_steps = temp
        return current_state
