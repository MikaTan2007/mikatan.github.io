import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
G = 9.81
L1, L2 = 1.0, 1.0
M1, M2 = 1.0, 1.0

def equations(t, y):
    th1, w1, th2, w2 = y
    
    # Mass matrix and force vector for double pendulum
    # (Derived from Euler-Lagrange equations)
    delta = th2 - th1
    den1 = (M1 + M2) * L1 - M2 * L1 * np.cos(delta) * np.cos(delta)
    
    dth1 = w1
    dw1 = (M2 * L1 * w1**2 * np.sin(delta) * np.cos(delta) +
           M2 * G * np.sin(th2) * np.cos(delta) +
           M2 * L2 * w2**2 * np.sin(delta) -
           (M1 + M2) * G * np.sin(th1)) / den1

    dth2 = w2
    dw2 = (-M2 * L2 * w2**2 * np.sin(delta) * np.cos(delta) +
           (M1 + M2) * (G * np.sin(th1) * np.cos(delta) - 
           L1 * w1**2 * np.sin(delta) - G * np.sin(th2))) / (L2 / L1 * den1)
    
    return [dth1, dw1, dth2, dw2]

# Integrate
y0 = [np.pi/2, 0, np.pi/2, 0] # Initial angles and velocities
t_span = (0, 50)
t_eval = np.linspace(0, 50, 10000)
sol = solve_ivp(equations, t_span, y0, t_eval=t_eval)

# Plotting the "Phase Projection"
plt.figure(figsize=(8, 8))
plt.plot(sol.y[1], sol.y[3], lw=0.5, color='black') # w1 vs w2
plt.title("Double Pendulum Phase Projection ($\omega_1$ vs $\omega_2$)")
plt.xlabel("$\omega_1$")
plt.ylabel("$\omega_2$")
plt.axis('off')
plt.show()