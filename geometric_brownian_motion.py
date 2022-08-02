from __future__ import annotations
import logging

from brownian_motion import DiscreteWienerProcess
from numpy import array, arange, exp


class DiscreteGeometricBrownianMotion(DiscreteWienerProcess):
    """
    Class representing a discrete Geometric Brownian Motion process. If `cov_matrix` is not None,
    then the process is deemed multivariate. Otherwise `delta_t` must be non None, it will be used
    as a variance of a univariate process.
    """

    def __init__(self,
                 mu: array | list,
                 sigma: array | list,
                 initial_position: float | array | list = 100,
                 delta_t: float = 1,
                 cov_matrix: array | list = None):
        """

        Parameters
        ----------
        mu: 1D array-like, parameter mu in the process (for each scalar variable in a vector variable)
        sigma: 1D array-like, parameter sigma in the process (for each scalar variable in a vector variable)
        initial_position: 1D array-like or float, the starting position of the process, either for each
        scalar variable (if array-like given) or a scalar number used for all scalar variables
        delta_t: float, delta t
        cov_matrix: 2D array-like, covariance matrix of a Wiener process (if multivariate). If not given,
        the process is deemed to be univariate
        """
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.info(f'Creating an instance of {self.logger.name}')

        super().__init__(initial_position=0,
                         delta_t=delta_t,
                         cov_matrix=cov_matrix)

        self.initial_position = initial_position
        self.mu = array(mu)
        self.sigma = array(sigma)

        self.delta_t = delta_t

        self.w_steps = self.steps
        self.w_paths = self.paths

        self.drift = self.mu - .5 * self.sigma ** 2
        self.diffusion = None

        self.cumulative_drift = None
        self.cumulative_diffusion = None  # full(shape=(n,), fill_value=nan)

    def _generate_cumulative_drift(self,
                                   n: int) -> array:
        """
        Calculate a cumulative drift of the process of length n
        Parameters
        ----------
        n: int, the # of discrete process states

        Returns
        -------
        2D numpy.array, the cumulative drift for each variable (each column represents a variable drift)
        """
        cumulative_drift = [self.drift] * array([arange(n) for _ in range(self._dimension)]).T * self.delta_t
        return cumulative_drift

    def _generate_cumulative_diffusion(self,
                                       w_path: array) -> array:
        cumulative_diffusion = [self.sigma] * w_path * self.delta_t ** .5
        return cumulative_diffusion

    def _generate_geom_path(self,
                            cumulative_drift: array,
                            cumulative_diffusion: array) -> array:
        path = [self.initial_position] * exp(cumulative_drift + cumulative_diffusion)
        return path

    def generate_paths(self,
                       n: int,
                       n_samples: int = 1):
        """
        Generate `n_samples` paths of the process, each of length `n`

        Parameters
        ----------
        n: `int`, number of process states in each sample. Note that it is by 1 larger than #
        of steps. Thus, for instance, if n = 100, the first state will be according to `initial_position` and 99
        random states will be generated
        n_samples: `int`, number of paths
        """
        self.n = n
        self.n_samples = n_samples

        initial_position_geom = self.initial_position  # workaround to save ini pos for geometric process
        self.initial_position = [0 for _ in range(self._dimension)]
        super().generate_paths(n,
                               n_samples=n_samples)
        self.initial_position = initial_position_geom

        self.w_steps = self.steps.copy()
        self.w_paths = self.paths.copy()
        self.steps = None

        self.cumulative_drift = [self._generate_cumulative_drift(n) for _ in range(self.n_samples)]

        self.diffusion = [self._generate_cumulative_diffusion(_w_steps) for _w_steps in self.w_steps]
        self.cumulative_diffusion = [self._generate_cumulative_diffusion(_w_steps) for _w_steps in self.w_paths]

        self.paths = [self._generate_geom_path(_cumulative_drift, _cumulative_diffusion)
                      for (_cumulative_drift, _cumulative_diffusion)
                      in zip(self.cumulative_drift, self.cumulative_diffusion)]

        self.cumulative_drift = array(self.cumulative_drift).copy()
        self.diffusion = array(self.diffusion).copy()
        self.cumulative_diffusion = array(self.cumulative_diffusion).copy()
        self.paths = array(self.paths).copy()
