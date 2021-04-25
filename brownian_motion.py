from scipy import stats as sts
import numpy as np


class DiscreteWienerProcess:
    def __init__(self, n=100, initial_position=0):
        self.n = n
        self.initial_position = initial_position
        self.steps = None
        self.path = None

    def generate_path(self):
        self.steps = sts.norm(loc=0, scale=1).rvs(size=self.n - 1)
        self.path = np.append(self.initial_position, self.steps.cumsum() + self.initial_position)


class DiscreteWienerProcessSampling(DiscreteWienerProcess):
    def __init__(self, n=100, initial_position=0, n_samples=10):
        DiscreteWienerProcess.__init__(self, n=n, initial_position=initial_position)
        self.n_samples = n_samples
        self.steps_list = np.full(shape=(n_samples, n - 1), fill_value=np.nan)
        self.paths = np.full(shape=(n_samples, n), fill_value=np.nan)

    def generate_paths(self):
        for i in range(self.n_samples):
            self.generate_path()
            self.steps_list[i] = self.steps
            self.paths[i] = self.path
