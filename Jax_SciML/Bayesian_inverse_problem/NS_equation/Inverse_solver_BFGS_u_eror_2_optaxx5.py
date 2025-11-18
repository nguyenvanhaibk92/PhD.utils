import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

import jax
jax.config.update("jax_enable_x64", True)

from jax.example_libraries import stax, optimizers
import jax.numpy as jnp
from jax import grad, value_and_grad, vmap, random, jit, lax

import jax.numpy as jnp

from jax.nn.initializers import glorot_normal, normal, zeros, ones

import pandas as pd
import numpy as np
import optax

# ## 0. Regularization parameters to test
MCDNN_AUG_list = [0, 10, 100, 500, 1000, 1500, 2200, 10000, 2e4,3e4, 4e4, 50000, 1e5, 2e5,5e5 ]

# Choose which sample to process (0-indexed)
SAMPLE_INDEX = 0  # Change this to select different samples

# Create array to store both alpha and error: [alpha, error] for each test
results = np.zeros((len(MCDNN_AUG_list), 2))  # 2 columns: alpha, error

# Create arrays to store inverted parameters for each alpha value
inverted_parameters = np.zeros((len(MCDNN_AUG_list), 24))  # 24 parameters based on num_truncated_series

# Variables to track the best result
best_error = float('inf')
best_alpha = None
best_inverted_params = None

case = 0

# key_noise_train, key_noise_test = random.split(random.PRNGKey(2))
key_noise_train = random.PRNGKey(case)
key_noise_test = random.PRNGKey(case)

Net_work_key = 100

# ## 1. Loading data by pandas
# Neural network models
num_train = 10000
num_test = 500

# Physical model information
n = 32
num_observation = 20  # number of observed points
dimension_of_PoI = (n)**2  # external force field
num_truncated_series = 24

# train_input_file_name = 'Train_Observations' + str(num_train)
# train_output_file_name = 'Train_PoI_samples_d' + str(num_train)
test_input_file_name = 'Test_Observations' + str(num_test)
test_output_file_name = 'Test_PoI_samples_d' + str(num_test)

# df_train_Observations = pd.read_csv('data/' + train_input_file_name + '.csv')
# df_train_Parameters = pd.read_csv('data/' + train_output_file_name + '.csv')
df_test_Observations = pd.read_csv('data/' + test_input_file_name + '.csv')
df_test_Parameters = pd.read_csv('data/' + test_output_file_name + '.csv')

# train_Observations_synthetic = np.reshape(df_train_Observations.to_numpy(), (num_train, -1))
# train_Parameters = np.reshape(df_train_Parameters.to_numpy(), (num_train, -1))
test_Observations_synthetic = np.reshape(df_test_Observations.to_numpy(), (num_test, -1))
test_Parameters = np.reshape(df_test_Parameters.to_numpy(), (num_test, -1))

# ### 1.1 Add noise
noise_level = 0.01
# train_Observations = train_Observations_synthetic + noise_level * (random.normal(key_noise_train, shape=train_Observations_synthetic.shape)) * jnp.reshape(jnp.max(train_Observations_synthetic, axis=1), (num_train, 1))
test_Observations = test_Observations_synthetic + noise_level * (random.normal(key_noise_test, shape=test_Observations_synthetic.shape)) * jnp.reshape(jnp.max(test_Observations_synthetic, axis=1), (num_test, 1))

# Extract single sample
single_test_observation = test_Observations[SAMPLE_INDEX, :]
single_test_parameter = test_Parameters[SAMPLE_INDEX, :]

# ### 1.2 Load Basis functions and observation locations
df_Basis = pd.read_csv('data/Basis' + '.csv')
df_obs_locations = pd.read_csv('data/obs_locations' + '.csv')

Basis = np.reshape(df_Basis.to_numpy(), (dimension_of_PoI, num_truncated_series))
obs_locations = np.reshape(df_obs_locations.to_numpy(), (num_observation))

Basis = jax.numpy.asarray(Basis)
obs_locations = jax.numpy.asarray(obs_locations)

# ## 2 - Spectral method for 2D Navier-Stokes equation
# initialize parameters
N = n

# steps size
dx = 1 / N
dy = 1 / N

x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

visc = 1e-3
T_end = 10.0
delta_t = 1e-2 * 5

# Maximum frequency
k_max = jnp.floor(N/2.0)

# Number of steps to final time
steps = jnp.ceil(T_end/delta_t).astype(int)

# force source
f = 0.1*(jnp.sin(2*jnp.pi*(X + Y)) + jnp.cos(2*jnp.pi*(X + Y)))

# Forcing to Fourier space
f_h = jnp.fft.fft2(f)

# Wavenumbers in y-direction
k = jnp.concatenate((jnp.arange(0, k_max, 1, dtype=int), jnp.arange(-k_max, 0, 1, dtype=int)))
k_y = jnp.tile(k, (32, 1))

# Wavenumbers in x-direction
k_x = k_y.T

# Negative Laplacian in Fourier space
lap = 4*(np.pi**2)*(k_x**2 + k_y**2)
lap = lap.at[0, 0].set(1.0)

# Dealiasing mask
dealias = jnp.logical_and(jnp.abs(k_y) < 2. / 3. * k_max, jnp.abs(k_x) < 2. / 3. * k_max) * 1

@jit
def body_loop(i, w_h):
    # Stream function in Fourier space: solve Poisson equation
    psi_h_real = jnp.real(w_h)/lap
    psi_h_imag = jnp.imag(w_h)/lap
    
    # Velocity field in x-direction = psi_y
    q_real, q_imag = psi_h_real, psi_h_imag
    temp = q_real
    
    q_real = -2 * jnp.pi * k_y * q_imag
    q_imag = 2 * jnp.pi * k_y * temp
    q = jnp.real(jnp.fft.ifft2(q_real + q_imag * 1j))
    
    # Velocity field in y-direction = -psi_x
    v_real, v_imag = psi_h_real, psi_h_imag
    temp = v_real
    
    v_real = 2 * jnp.pi * k_x * v_imag
    v_imag = -2 * jnp.pi * k_x * temp
    v = jnp.real(jnp.fft.ifft2(v_real + v_imag * 1j))
    
    # Partial x of vorticity
    w_x_real, w_x_imag = jnp.real(w_h), jnp.imag(w_h)
    temp = w_x_real
    w_x_real = -2 * jnp.pi * k_x * w_x_imag
    w_x_imag = 2 * jnp.pi * k_x * temp
    w_x = jnp.real(jnp.fft.ifft2(w_x_real + w_x_imag * 1j))
    
    # Partial y of vorticity
    w_y_real, w_y_imag = jnp.real(w_h), jnp.imag(w_h)
    temp = w_y_real
    w_y_real = -2 * jnp.pi * k_y * w_y_imag
    w_y_imag = 2 * jnp.pi * k_y * temp
    w_y = jnp.real(jnp.fft.ifft2(w_y_real + w_y_imag * 1j))
    
    # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
    F_h = jnp.fft.fft2(q*w_x + v*w_y)
    
    # Dealias
    F_h = dealias * F_h
    
    # Cranck-Nicholson update
    w_h = (-delta_t*F_h + delta_t*f_h + (1.0 - 0.5*delta_t*visc*lap)*w_h)/(1.0 + 0.5*delta_t*visc*lap)

    return w_h

# Forward-solver
def solve_forward(x):
    # ==================== for each sample ================== #
    w0 = jnp.asarray(jnp.reshape((Basis @ x.T), (N, N)))

    # Initial vorticity to Fourier space
    w_h = jnp.fft.fft2(w0)
    
    w_h = jax.lax.fori_loop(0, steps, body_loop, (w_h))
    
    # Transform back to physical space
    w = jnp.real(jnp.fft.ifft2(w_h))
    
    y_obs = (w.flatten())[[obs_locations],]
    return y_obs.flatten()

def misfit(x, y, alpha):
    misfit = jnp.linalg.norm(solve_forward(x) - y)**2
    regulaziration = jnp.linalg.norm(x)**2
    
    return alpha * misfit + regulaziration

from jax._src.scipy.optimize.minimize import minimize as minimize
@jit
def solve_BFGS(y, alpha):
    x_0 = 0.01 * jax.random.normal(random.PRNGKey(case), single_test_parameter.shape)
    return (minimize(misfit, x_0, args = (y, alpha), method = 'BFGS', tol = 1e-16)).x

# def solve_BFGS(y, alpha):    
#     def f(x): 
#         return misfit(x, y, alpha)
    
#     solver = optax.lbfgs()
#     x_0 = 0.01 * jax.random.normal(random.PRNGKey(case), single_test_parameter.shape)
#     opt_state = solver.init(x_0)
#     value_and_grad = jax.jit(optax.value_and_grad_from_state(f))
    
#     for _ in range(50):
#         value, grad = value_and_grad(x_0, state=opt_state)
#         updates, opt_state = solver.update(grad, opt_state, x_0, value=value, grad=grad, value_fn=f)
#         x_0 = optax.apply_updates(x_0, updates)
#         # print('Objective function: ', f(x_0), np.linalg.norm(x_0))
#     return x_0
    
@jit
def acc(preds, true):
    return jnp.mean(jnp.square(Basis @ (preds - true).T)) / jnp.mean(jnp.square(Basis @ true.T))
    # return jnp.mean(jnp.square((preds - true))) / jnp.mean(jnp.square(true))

print(f"Processing sample {SAMPLE_INDEX}")
print("="*50)

for iii in range(len(MCDNN_AUG_list)):
    alpha_value = MCDNN_AUG_list[iii]
    
    # Solve for single sample
    x_single = solve_BFGS(single_test_observation, alpha_value)
    
    # Calculate error for single sample
    Err = acc(x_single, single_test_parameter)
    
    # Store inverted parameters
    inverted_parameters[iii, :] = x_single
    
    # Track the best result (lowest error)
    if Err < best_error:
        best_error = Err
        best_alpha = alpha_value
        best_inverted_params = x_single.copy()
    
    # Calculate losses for single sample
    Loss_begin = misfit(x_single*0, single_test_observation, alpha_value)
    Loss_end = misfit(x_single, single_test_observation, alpha_value)
    
    print("Case : %2d, Sample: %3d, Error : %.4f, alpha: %1.E, Loss_begin: %.2E, Loss_End: %.2E" % 
            (case, SAMPLE_INDEX, Err, alpha_value, Loss_begin, Loss_end))
    
    # Store both alpha and error
    results[iii, 0] = alpha_value  # First column: regularization parameter
    results[iii, 1] = Err          # Second column: inverse error
    
    # Optional: Print additional details for the single sample
    print(f"  Predicted parameters: {x_single}")
    print(f"  True parameters: {single_test_parameter}")
    print(f"  Parameter difference: {x_single - single_test_parameter}")
    print("-"*30)

# Save results with header
filename = f'Single_Sample_{SAMPLE_INDEX}_Alpha_Error_Results_case_{case}_Noise_{noise_level}.txt'
header = "Alpha,Error"
np.savetxt(filename, results, fmt='%1.11e', delimiter=',', header=header, comments='')

# Save best result summary
summary_filename = f'Single_Sample_{SAMPLE_INDEX}_Best_Summary_case_{case}_Noise_{noise_level}_inverse_solution'
np.save(summary_filename, best_inverted_params)

print(f"\nBest result:")
print(f"Alpha: {best_alpha}")
print(f"Error: {best_error}")
print(f"Best inverted parameters saved to: {summary_filename}.npy")