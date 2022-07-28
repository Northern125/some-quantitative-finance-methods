from scipy.stats import norm
from numpy import append, array


class DiscreteWienerProcess:
    """
    Class representing a discrete Wiener process.
    """
    def __init__(self,
                 initial_position: float = 0,
                 delta_t: float = 1):
        self.initial_position = initial_position
        self.delta_t = delta_t

        self.n = None
        self.n_samples = None

        self.steps = None
        self.paths = None

    def _generate_steps(self,
                        n_steps: int) -> array:
        """
        :param n_steps: `int`, number of steps
        :return: `numpy.array`, steps at each delta t
        """

        steps = norm(loc=0, scale=self.delta_t ** .5).rvs(size=n_steps)
        return steps

    def _generate_path(self,
                       steps: array) -> array:
        """
        Generate a random path given steps (increments at each delta t). Resulting array will contain 1 more element
        than `steps` array. For instance, if 100 steps are given, the resulting array will have 101 process states
        :param steps: `numpy.array`, generated steps (at each delta t)
        :return: `numpy.array`, discrete consequent process states
        """

        path = append(self.initial_position, steps.cumsum() + self.initial_position).copy()
        return path

    def generate_paths(self,
                       n: int,
                       n_samples: int = 1):
        """
        :param n: `int`, number of process states in each sample. Note that it is by 1 larger than # of steps. Thus,
        for instance, if n = 100, the first state will be according to `initial_position`
        and 99 random states will be generated
        :param n_samples: `int`, number of paths
        """
        self.n = n
        self.n_samples = n_samples

        self.steps = [self._generate_steps(self.n - 1) for _ in range(n_samples)]
        self.paths = [self._generate_path(_path_steps) for _path_steps in self.steps]

        self.steps = array(self.steps)
        self.paths = array(self.paths)
