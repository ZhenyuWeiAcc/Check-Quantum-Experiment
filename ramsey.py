#%%
import numpy as np
import matplotlib.pyplot as plt
def drho_dt(M):
    """This describes decoherence from environment. Linblad equation"""
    gamma = 0.06
    miu_B = 0.5
    b11 = gamma * M[0,0]
    b12 = (-gamma / 2 + miu_B * 1j)*M[0,1]
    b21 = (-gamma / 2 + miu_B * 1j)*M[1,0]
    b22 = -gamma * M[1,1]
    rate = np.matrix([[b11,b12],[b21,b22]])
    return rate
def re_norm(M):
    """re normalize the matrix"""
    n = M[0,0] + M [1,1]
    A = M / n
    return A
# source:
# https://quantumcomputing.stackexchange.com/questions/35507/ramsey-measurement-for-characterising-a-qubit
# aim: just check the math of Ramsey experiment
#%%
N_measure = 1000
state_i = [[0],[1]] #initialized in |1> 
theta = np.pi / 2
X_pi_2 = np.matrix([[np.cos(theta / 2)+0*1j, 0 - np.sin(theta / 2)*1j], [0 - np.sin(theta / 2)*1j, np.cos(theta / 2) + 0*1j]]) #rotate about x for pi/2 in block sphere
# %%
P_ = []
P_m = []
P_m_error = []
P__ = []
P__m = []
P__m_error = []
for t in range(1000):
    state = np.dot(X_pi_2, state_i)
    a11 = np.dot(np.conj(state[0]) , state[0])
    a12 = np.dot(np.conj(state[0]) , state[1])
    a21 = np.dot(np.conj(state[1]) , state[0])
    a22 = np.dot(np.conj(state[1]) , state[1])
    density_matrix  = np.matrix([[a11[0,0],a12[0,0]],[a21[0,0],a22[0,0]]])

    dt = 5*10**(-2)

    for i in range(t):
        # time evolving
        density_matrix = density_matrix + drho_dt(density_matrix) * dt

    density_matrix = np.dot(X_pi_2 , density_matrix)
    density_matrix = np.dot(density_matrix, X_pi_2.H)
    density_matrix = re_norm(density_matrix)
    P_ = np.append(P_, np.abs(density_matrix[1,1]))
    P__ = np.append(P__, np.abs(density_matrix[0,0]))
    if t % 10 == 0:
        P_m = np.append(P_m, np.random.binomial(N_measure,np.abs(density_matrix[1,1])) / N_measure)
        P_m_error = np.append(P_m_error, np.sqrt(density_matrix[1,1]*(1 - density_matrix[1,1])) / np.sqrt(N_measure))

        P__m = np.append(P__m, np.random.binomial(N_measure,np.abs(density_matrix[0,0])) / N_measure)
        P__m_error = np.append(P__m_error, np.sqrt(density_matrix[0,0]*(1 - density_matrix[0,0])) / np.sqrt(N_measure))
        

# %%
plt.figure(dpi=800)
plt.xlabel('Time [a.u.]')
plt.ylabel('Probability')
plt.plot(range(t+1), P_, label = 'state 0', linewidth = 0.5)
plt.errorbar(np.arange(0,t+1,10), P_m, yerr = P_m_error, fmt='r.', label = 'state 0 measurement')
plt.plot(range(t+1), P__, label = 'state 1', linewidth = 0.5)
plt.errorbar(np.arange(0,t+1,10), P__m, yerr = P__m_error, fmt='b.', label = 'state 1 measurement')
plt.legend()
# %%
