from __future__ import annotations
import logging

from scipy.stats import multivariate_normal
from numpy import append, array


class DiscreteWienerProcess:
    """
    Class representing a discrete Wiener process. If `cov_matrix` is not None, then the process is deemed multivariate.
    Otherwise `delta_t` must be non None, it will be used as a variance of a univariate process.
    """
    def __init__(self,
                 initial_position: float | array | list = 0,
                 delta_t: float = 1,
                 cov_matrix: array | list = None):
        """
        :param initial_position: `float`, initial state of the process
        :param delta_t: `float`, time interval between each process state. In general case units can be any, they only
        need to be equal among all the variables. Year fraction is preferable as it is a world standard
        :cov_matrix: 2D array-like or None, the covariance matrix of the process, if it's multivariate
        """
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.info(f'Creating an instance of {self.logger.name}')

        if cov_matrix is None and delta_t is not None:
            self.logger.info('`cov_matrix` is None, the process is univariate')

            self.cov_matrix = delta_t
            self._dimension = 1

        elif cov_matrix is not None:
            self.logger.info('`cov_matrix` is not None, the process is multivariate')

            cov_matrix = array(cov_matrix)

            if len(cov_matrix.shape) != 2:
                raise ValueError(f'`cov_matrix` shape must be 2D, got {len(cov_matrix.shape)}D')
            if cov_matrix.shape[0] != cov_matrix.shape[1]:
                raise ValueError('`cov_matrix` is not square')

            self.cov_matrix = cov_matrix
            self._dimension = cov_matrix.shape[0]
        else:
            raise ValueError('At least one of the 2 parameters `delta_t` and `cov_matrix` must be non None')

        if type(initial_position) is float or type(initial_position) is int:
            self.initial_position = array([initial_position for _ in range(self._dimension)])
        elif type(self.initial_position) is list or type(self.initial_position) is array \
                and len(self.initial_position) == self._dimension:
            self.initial_position = array(initial_position)
        else:
            raise ValueError('`self.initial_position` must be 1D array-like with length equal to covariance matrix '
                             '`shape[0]`')

        self.n = None
        self.n_samples = None

        self.steps = None
        self.paths = None

    def _generate_steps(self,
                        n_steps: int) -> array:
        """
        :param n_steps: `int`, number of steps
        :return: 2D `numpy.array`, steps at each delta t
        """
        steps = multivariate_normal(mean=[0 for _ in range(self._dimension)], cov=self.cov_matrix).rvs(size=n_steps)
        if self._dimension == 1 or type(steps) is float:
            steps = array([steps]).T.copy()
        return steps

    def _generate_path(self,
                       steps: array) -> array:
        """
        Generate a random path given steps (increments at each delta t). Resulting array will contain 1 more element
        than `steps` array. For instance, if 100 steps are given, the resulting array will have 101 process states
        :param steps: `numpy.array`, generated steps (at each delta t)
        :return: 2D `numpy.array`, discrete consequent process states
        """
        path = append([self.initial_position], steps.cumsum(axis=0) + [self.initial_position], axis=0).copy()
        return path

    def generate_paths(self,
                       n: int,
                       n_samples: int = 1):
        """
        Generate `n_samples` paths of the process, each of length `n`

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
