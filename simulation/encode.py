
import numpy as np
import math
import scipy.signal
import scipy.io
from scipy import special
import os
from scipy.fftpack import fft, ifft
import acousticTrackingDataset as at_dataset
import scipy.special as sp
import scipy.io as sio
from math import factorial

array_setup = at_dataset.eigenmike_array_setup

def HOAencode(x, fs, order=1, coef=3e-2, channel_index=[5, 9, 25, 21], rm = 1):
    """
    channel 6, 10, 26, 22
    (45, 35), (45, -35), (135, -35), (-135, 35) [azimuth(-180, 180), elevation(-90, 90)]
    (55, 45), (125, 315), (125, 135), (55, 225) [theta(0 - 90), phi(0 - 360)] define in eigenmike
    """
    # print('hoa x', x.shape)
    channel = x[0, :].size
    w = scipy.signal.windows.hann(1025)
    w = w[:1024]
    wcopy = np.transpose(np.tile(w, (channel, 1)))
    # print('w', w.shape)
    # print('wcopy', wcopy.shape)
    M = w.size
    N = 1024 
    H = N // 2
    hM1 = (M + 1)//2
    hM2 = M // 2
    x = np.vstack((np.zeros((hM2, channel)), x))
    x = np.vstack((x, np.zeros((hM1, channel))))  # padding
    pin = hM1
    pend = x[:,0].size - hM1
    Y = enmatrixY(order, channel_index)
    # print(Y.T)
    # print(Y.T / np.sqrt(np.pi * 4))
    E = np.linalg.pinv(Y)
    f = np.linspace(0, fs / 2, N // 2 + 1)
    Eq = matrixEQ(order, f, coef=coef, rm = rm)
    # print(Eq.shape)
    x_HOA = np.zeros((x[:, 0].size, (order + 1) ** 2))
    while pin <= pend:
        x1 = x[pin - hM1: pin + hM2, :]
        x1 = np.multiply(x1, wcopy)
        X1 = np.transpose(fft(np.transpose(x1), N))
        X1_HOA = np.zeros((X1[:, 0].size, (order + 1) ** 2), dtype=complex)
        # print(X1.shape)
        # print(N // 2 + 1)
        # print(E.shape)
        temp = np.dot(X1[: N // 2 + 1, :], E)
        # print('temp', temp.shape)
        X1_HOA[: N // 2 + 1, :] = np.multiply(temp, Eq)
        for i in range(1, N // 2):
            X1_HOA[N // 2 + i, :] = np.conjugate(X1_HOA[N // 2 - i, :])
        X1_HOA[0, :] = X1_HOA[0, :].real
        X1_HOA[N // 2, :] = X1_HOA[N // 2, :].real
        x1_HOA = (np.transpose(ifft(np.transpose(X1_HOA)))).real
        x_HOA[pin - hM1: pin + hM2,:] += x1_HOA
        pin += H
    x_HOA = np.delete(x_HOA, range(hM2), 0)
    x_HOA = np.delete(x_HOA, range(x_HOA[:, 0].size - hM1, x_HOA[:, 0].size), 0)
    return x_HOA, fs

# def enmatrixY(order, channel_index):
#     Y = scipy.io.loadmat(os.path.join('.', 'eigenmike-Y.mat'))['Y']
#     return Y[:(order + 1) ** 2, channel_index]

def cart2sph(cart):
	xy2 = cart[:,0]**2 + cart[:,1]**2
	sph = np.zeros_like(cart)
	sph[:,0] = np.sqrt(xy2 + cart[:,2]**2)
	sph[:,1] = np.arctan2(np.sqrt(xy2), cart[:,2]) # Elevation angle defined from Z-axis down
	sph[:,2] = np.arctan2(cart[:,1], cart[:,0])
	# print(cart)
	# print(sph)
	return sph

def enmatrixY(order, channel_index):
    if os.path.exists(os.path.join('.', 'eigenmike-Y.mat')):
        Y = scipy.io.loadmat(os.path.join('.', 'eigenmike-Y.mat'))['Y']
        # print(Y.shape)
        return Y[:(order + 1) ** 2, channel_index]
    else:
        mic_pos = array_setup.mic_pos
        # print(mic_pos)
        mic_sph = cart2sph(mic_pos)
        # print(mic_sph[:, -1:0:-1])
        # mic_sph = np.zeros_like(mic_sph)
        # with open('gs_data1.txt', 'r') as f:
        #     lines = [list(map(float, line.strip('\n').split('\t'))) for line in f.readlines()]
        # data = np.array(lines)
        # mic_sph[:, 1] = np.pi / 2 - data[:, -1]
        # mic_sph[:, 2] = data[:, 0]
        mic_number = mic_sph.shape[0]
        y = np.zeros((mic_number, (order+1) ** 2))
        for mic_ii in range(mic_number):
            # print(mic_sph[mic_ii,2], mic_sph[mic_ii,1])
            sh = getSH(order, mic_sph[mic_ii,2], mic_sph[mic_ii,1], 'real')
            # print(sh)
            y[mic_ii,:] = sh
        data = {'Y': y.T}
        sio.savemat(os.path.join('.', 'eigenmike-Y.mat'), data)
    return y.T

def legendre(n, x):
    (temp,_) = sp.lpmn(n, n, x)
    return temp[:,n]

def getSH(N, azi, inc, basisType):
    # azi, inc : (Ndirs,)
    # basisType: select from {'complex', 'real'}
    azi = np.array(azi).reshape(-1,)
    inc = np.array(inc).reshape(-1,)

    Ndirs = azi.shape[0]
    Nharm = (N+1)**2

    Y_N = np.zeros((Nharm, Ndirs))
    idx_Y = 0
    for n in range(N+1):

        m = np.arange(n+1)

        # complex type of getSH
        if basisType == 'complex':
            Y_N = Y_N.astype('complex64')

            Lnm = np.zeros((Ndirs, n+1))
            for idir in range(Ndirs):
                Lnm[idir, :] = legendre(n, np.cos(inc[idir]))    ### bug!
            Lnm = np.transpose(Lnm)

            norm = np.zeros((n+1, 1))
            for i in m:
                norm[i,:] = np.sqrt((2*n+1)*factorial(n-i) / (4*np.pi*factorial(n+i)))

            Nnm = np.matmul(norm, np.ones((1,Ndirs)))
            
            Exp = np.exp((1j) * np.matmul(m.reshape(-1,1), azi.reshape(1,-1)))
            Ynm_pos = Nnm * Lnm * Exp

            if n != 0:
                condon = np.matmul((-1) ** m[-1:0:-1].reshape(-1,1), np.ones((1, Ndirs)))
                Ynm_neg = condon * np.conj(Ynm_pos[-1:0:-1,:])
            else:
                Ynm_neg = np.zeros((0,Ynm_pos.shape[1]))
            
            Ynm = np.vstack([Ynm_neg, Ynm_pos])

        # real type of getSH
        elif basisType == 'real':
            Lnm_real = np.zeros((Ndirs, n+1))
            for idir in range(Ndirs):
                Lnm_real[idir, :] = legendre(n, np.cos(inc[idir]))    ### bug!
            Lnm_real = np.transpose(Lnm_real)
            if n != 0:
                temp = np.vstack([m[-1:0:-1].reshape(-1,1), m.reshape(-1,1)])
                condon = np.matmul((-1) ** temp, np.ones((1, Ndirs)))
                Lnm_real = condon * np.vstack([Lnm_real[-1:0:-1,:], Lnm_real])

            norm_real = np.zeros((n+1, 1))
            for i in m:
                norm_real[i,:] = np.sqrt((2*n+1)*factorial(n-i) / (4*np.pi*factorial(n+i)))
            Nnm_real = np.matmul(norm_real, np.ones((1,Ndirs)))
            if n != 0:
                Nnm_real = np.vstack([Nnm_real[-1:0:-1,:], Nnm_real])

            CosSin = np.zeros((2*n+1, Ndirs))
            CosSin[n,:] = np.ones((1,Ndirs))
            if n != 0:
                CosSin[m[1:]+n,:] = np.sqrt(2) * np.cos(np.matmul(m[1:].reshape(-1,1), azi.reshape(1,-1)))
                CosSin[-m[-1:0:-1]+n,:] = np.sqrt(2) * np.sin(np.matmul(m[-1:0:-1].reshape(-1,1), azi.reshape(1,-1)))
            Ynm = Nnm_real * Lnm_real * CosSin

        Y_N[idx_Y:idx_Y+(2*n+1), :] = Ynm
        idx_Y = idx_Y + 2*n+1
    
    Y_N = np.transpose(Y_N)
    return Y_N

def matrixEQ(order, f, coef, rm = 1):
    """
    order: encoding order
    f: array of frequency
    coef: regularization coefficient of radial function
    return:
    EQ: x * (N+1)^2
    """
    r = 42e-3 * rm
    veq = np.vectorize(eq)
    EQ = np.zeros((np.size(f),(order+1)**2),dtype=complex)
    for i in range(order+1):
        for j in range(2*i+1):
            EQ[:,i**2+j] = veq(f,r,i,coef)        
    return EQ
            
def eq(x, r, n,coef):
    """
    x: single frequency
    r: array radius
    n: order
    coef: regularization coefficient of radial function
    """
    kr = 2 * np.pi * x * r / 340 + (x == 0) * np.finfo(float).eps
    hnde = math.sqrt(np.pi / (2 * kr)) * (n / kr * special.hankel2(n + 1 / 2, kr)-special.hankel2(n + 3 / 2, kr))
    bn = (-1j)**(n+1)*(-1)**n/(kr**2*hnde)
    out = np.conjugate(bn)/((abs(bn))**2 + coef**2)
    return out

