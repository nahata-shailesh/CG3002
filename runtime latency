import numpy as np
import time 
from scipy.stats import scoreatpercentile

n_instances = XTest.shape[0]
runtimes = np.zeros(n_instances, dtype=np.float)
for i in range(n_instances):
    instance = XTest[[i], :]
    start = time.time()
    rfc.predict(instance)
    runtimes[i] = time.time() - start #time.time() return time in seconds

print("atomic_benchmark runtimes:", min(runtimes), scoreatpercentile(
        runtimes, 50), max(runtimes))
