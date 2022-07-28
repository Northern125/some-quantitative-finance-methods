from scipy import stats as sts
import numpy as np


class DiscreteWienerProcess:
    """
    Class representing a discrete Wiener process.
    """
    def __init__(self,
                 # n: int = 100,
                 initial_position: float = 0,
                 delta_t: float = 1):
        # self.n = n
        self.initial_position = initial_position
        self.steps = None
        self.path = None
        self.delta_t = delta_t
        self.n = None

    def generate_path(self,
                      n: int):
        """
        Generate a random path of length `n` (including starting position)
        :param n: int, the length of the path (# of discrete process states). It includes the starting point.
        Thus, for example, if `n = 100`, 99 states will be generated
        """
        self.n = n
        self.steps = sts.norm(loc=0, scale=self.delta_t ** .5).rvs(size=self.n - 1)
        self.path = np.append(self.initial_position, self.steps.cumsum() + self.initial_position).copy()


class DiscreteWienerProcessSampling(DiscreteWienerProcess):
    def __init__(self,
                 n: int = 100,
                 initial_position: float = 0,
                 n_samples: int = 10,
                 delta_t: float = 1):
        super().__init__(self, n=n, initial_position=initial_position, delta_t=delta_t)
        self.n_samples = n_samples
        self.steps_list = [np.full(shape=(n_samples, n - 1), fill_value=np.nan)]
        self.paths = np.full(shape=(n_samples, n), fill_value=np.nan)

    def generate_paths(self):
        for i in range(self.n_samples):
            self.generate_path()
            self.steps_list[i] = self.steps
            self.paths[i] = self.path
