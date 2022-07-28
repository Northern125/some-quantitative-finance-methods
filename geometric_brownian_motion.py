from brownian_motion import DiscreteWienerProcess
import numpy as np


class DiscreteGeometricBrownianMotion(DiscreteWienerProcess):
    def __init__(self, n=100, initial_position=100, mu=0, sigma=1):
        DiscreteWienerProcess.__init__(self, n=n, initial_position=0)

        self.initial_position = initial_position
        self.mu = mu
        self.sigma = sigma

        self.w_steps = self.steps
        self.w_path = self.path

        self._drift = self.mu - .5 * self.sigma ** 2
        self._diffusion = None

        self._cumulative_drift = self._drift * np.arange(self.n)
        self._cumulative_diffusion = np.full(shape=(n,), fill_value=np.nan)

    def generate_path(self):
        initial_position_temp = self.initial_position  # workaround to keep ini pos for geometric process
        self.initial_position = 0
        DiscreteWienerProcess.generate_path(self)
        self.initial_position = initial_position_temp

        self.w_steps = self.steps
        self.w_path = self.path

        self._diffusion = self.sigma * self.w_steps

        self._cumulative_diffusion = self.sigma * self.w_path

        self.path = self.initial_position * np.exp(self._cumulative_drift + self._cumulative_diffusion)


class DiscreteGeometricBrownianMotionSampling(DiscreteGeometricBrownianMotion):
    def __init__(self, n=100, initial_position=100, mu=0, sigma=1, n_samples=10):
        DiscreteGeometricBrownianMotion.__init__(self, n=n, initial_position=initial_position, mu=mu, sigma=sigma)

        self.n_samples = n_samples

        self.w_steps_list = np.full(shape=(n_samples, n - 1), fill_value=np.nan)
        self.w_paths = np.full(shape=(n_samples, n), fill_value=np.nan)

        self.steps_list = np.full(shape=(n_samples, n - 1), fill_value=np.nan)
        self.paths = np.full(shape=(n_samples, n), fill_value=np.nan)

        self._drifts = np.full(shape=(n_samples, n), fill_value=self._drift)
        self._diffusions = np.full(shape=(n_samples, n - 1), fill_value=np.nan)

        self._cumulative_drifts = np.full(shape=(n_samples, n), fill_value=self._cumulative_drift)
        self._cumulative_diffusions = np.full(shape=(n_samples, n), fill_value=np.nan)

    def generate_paths(self):
        for i in range(self.n_samples):
            self.generate_path()

            self.steps_list[i] = self.steps
            self.paths[i] = self.path

            self.w_steps_list = self.w_steps
            self.w_paths = self.w_path

            self._diffusions[i] = self._diffusion
            self._drifts[i] = self._drift

            self._cumulative_diffusions[i] = self._cumulative_diffusion
