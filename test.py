from sobi import *
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import time

plt.ion()

N = 10000                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
tt = np.linspace(0,200,N)
Strue = generate_random_signals(tt, 10)
Atrue = np.random.random([Strue.shape[0],Strue.shape[0]])
X = Atrue.dot(Strue)

start_time = time.time()

num_lags = 5000
S,A,W = sobi(X, num_lags = None, random_order=True, eps=1.0e-12)
elapsed_time = time.time()-start_time
print('\n elapsed_time: {} seconds.'.format(elapsed_time))

plot_signals_sorted = lambda x: plot_signals(sort_by_freq(x))

fig1,axlist1 = plot_signals_sorted(X)
plt.suptitle('Mixed Signals')

fig2, axlist2 = plot_signals_sorted(Strue)
plt.suptitle('True Signals')

fig3, axlist3 = plot_signals_sorted(S)
plt.suptitle('Estimated Signals')