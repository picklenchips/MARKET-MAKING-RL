import numpy as np
import stochastic.processes.noise as noise
GaussianNoise = noise.GaussianNoise

from stochastic.processes.continuous.brownian_motion import BrownianMotion


model = BrownianMotion()
new = model._sample_brownian_motion(1000)
print(new)


