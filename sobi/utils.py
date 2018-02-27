import numpy as np
import matplotlib.pyplot as plt
import random

def sort_by_freq(X):

    N = X.shape[1]
    f  = np.fft.fftfreq(N)
    Xf = np.fft.fft(X)
    
    Xf = Xf*Xf
    Xf = Xf/(Xf.sum(1)[:,None])

    avg_freq = Xf.dot(abs(f))

    idx = np.argsort(avg_freq)

    Xsorted = X[idx,:]*1.0
    
    return Xsorted



def autotile(x3d):

    num_tiles = x3d.shape[0]
    row_dim = int(np.sqrt(num_tiles))
    col_dim = int(np.ceil(num_tiles/row_dim))
    fig,axlist = plt.subplots(row_dim, col_dim, sharex=True, sharey=True)

    for i,ax in enumerate(axlist.ravel()):
        ax.imshow(x3d[i])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

def plot_signals(X):

    num_signals = X.shape[0]
    fig, axlist = plt.subplots(num_signals, 1, sharex=True, sharey=False)
    for i,ax in enumerate(axlist):
        ax.plot(X[i], linewidth=1.0)
        ax.yaxis.set_visible(False)

    return fig, axlist

def sin(tt, T=1, phi=0):

    f = 1.0/T
    y = np.sin(2*np.pi*f*(tt-phi))

    return y

def square(tt, T=1, phi=0):
    y = sin(tt, T, phi)
    y = np.sign(y)
    return y

def tri(tt, T=1, phi=0):
    y = square(tt, T, phi)
    y = np.cumsum(y)
    y = y/np.max(y)

    return y

def generate_random_signals(tt, num_signals):

    S = np.empty([num_signals, len(tt)])

    for i in range(num_signals):
        f = random.choice([sin, square, tri])
        T = random.random()*max(tt)/100
        phi = random.random()*max(tt)
        y = f(tt, T, phi)
        S[i,:] = y
    return S