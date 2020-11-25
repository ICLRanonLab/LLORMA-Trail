
import numpy as np
data = np.genfromtxt('ratings.dat',
                     dtype=None,
                     delimiter='::')
print(data)
count = np.shape(np.unique(data[:,1]))
print(count)