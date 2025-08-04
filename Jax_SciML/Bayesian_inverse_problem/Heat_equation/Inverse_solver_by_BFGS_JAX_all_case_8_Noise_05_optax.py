import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

import jax
jax.config.update("jax_enable_x64", True)

from jax.example_libraries import stax, optimizers
import jax.numpy as jnp
from jax import grad, value_and_grad, vmap, random, jit, lax

import jax.numpy as jnp

# from jax.nn.initializers import glorot_normal, normal, zeros, one

import pandas as pd
import numpy as np

from scipy import sparse
import optax


# ## 0. mcDNN-Encoder (0) or mcDNN-Decoder (1)?
MCDNN_AUG_list = [ 10, 1e2, 1e3, 2e3, 3e3,4e3, 5e3, 6e3,7e3, 8e3, 1e4, 1e5, 1e6]
# MCDNN_AUG_list = [0]

# Choose which sample to process (0-indexed)
SAMPLE_INDEX = 0  # Change this to select different samples

# Create array to store both alpha and error: [alpha, error] for each test
results = np.zeros((len(MCDNN_AUG_list), 2))  # 2 columns: alpha, error

# Create arrays to store inverted parameters for each alpha value
inverted_parameters = np.zeros((len(MCDNN_AUG_list), 15))  # Assuming 15 parameters based on num_truncated_series

# Variables to track the best result
best_error = float('inf')
best_alpha = None
best_inverted_params = None

case=0

# key_noise_train, key_noise_test =  random.split(random.PRNGKey(2))
key_noise_train = random.PRNGKey(case)
key_noise_test = random.PRNGKey(case)

Net_work_key = 100
# ## 1. Loading data by pandas
# Neural network models
num_train = 10000
num_test = 500


# train_input_file_name =  'poisson_2D_state_obs_train_o10_d' + str(num_train)+ '_n15_AC_1_1_pt5'
# train_output_file_name = 'poisson_2D_parameter_train_d' + str(num_train)+ '_n15_AC_1_1_pt5'
test_input_file_name =   'poisson_2D_state_obs_test_o10_d' + str(num_test)+ '_n15_AC_1_1_pt5'
test_output_file_name =  'poisson_2D_parameter_test_d' + str(num_test)+ '_n15_AC_1_1_pt5'


# df_train_Observations = pd.read_csv('data/' + train_input_file_name + '.csv') 
# df_train_Parameters = pd.read_csv('data/' + train_output_file_name + '.csv')
df_test_Observations = pd.read_csv('data/' + test_input_file_name + '.csv') 
df_test_Parameters = pd.read_csv('data/' + test_output_file_name + '.csv')


# train_Observations_synthetic = np.reshape(df_train_Observations.to_numpy(), (num_train,-1))
# train_Parameters = np.reshape(df_train_Parameters.to_numpy(), (num_train,-1))
test_Observations_synthetic = np.reshape(df_test_Observations.to_numpy(), (num_test,-1))
test_Parameters = np.reshape(df_test_Parameters.to_numpy(), (num_test,-1))

# ### 1.1 Add noise
noise_level = 0.01
# train_Observations = train_Observations_synthetic + noise_level * (random.normal(key_noise_train, shape=train_Observations_synthetic.shape)) * jnp.reshape(jnp.max(train_Observations_synthetic, axis = 1), (num_train, 1))
test_Observations = test_Observations_synthetic + noise_level * (random.normal(key_noise_test, shape=test_Observations_synthetic.shape)) * jnp.reshape(jnp.max(test_Observations_synthetic, axis = 1), (num_test, 1))

# Extract single sample
single_test_observation = test_Observations[SAMPLE_INDEX, :]
single_test_parameter = test_Parameters[SAMPLE_INDEX, :]

# train_set = jnp.hstack((train_Observations, train_Parameters))

# ### 1.2 Load Eigenvalue, Eigenvectors, observed indices, prematrices

# Physical model information
n = 15
dimension_of_PoI = (n + 1)**2 # external force field
num_observation = 10 # number of observed points
num_truncated_series = 15

df_Eigen = pd.read_csv('data/prior_mean_n15_AC_1_1_pt5' + '.csv') 
df_Sigma = pd.read_csv('data/prior_covariance_n15_AC_1_1_pt5' + '.csv') 

Eigen = np.reshape(df_Eigen.to_numpy(), (dimension_of_PoI, num_truncated_series))
Sigma = np.reshape(df_Sigma.to_numpy(), (num_truncated_series, num_truncated_series))

df_obs = pd.read_csv('data/poisson_2D_obs_indices_o10_n15' + '.csv')
obs_indices = np.reshape(df_obs.to_numpy(), (num_observation,-1))


boundary_matrix = sparse.load_npz('data/boundary_matrix_n15' + '.npz')
pre_mat_stiff_sparse = sparse.load_npz('data/prestiffness_n15' + '.npz')
load_vector_n15 = sparse.load_npz('data/load_vector_n15' + '.npz')
load_vector = sparse.csr_matrix.todense(load_vector_n15).T


df_free_index = pd.read_csv('data/prior_covariance_cholesky_n15_AC_1_1_pt5' + '.csv')
free_index = df_free_index.to_numpy()

i = 0
obs_transfered = []
for free_ind in free_index:
    if (free_ind in obs_indices):
        obs_transfered.append(i)
    i += 1

jjj = 0
obs_operator = np.zeros((num_observation, free_index.shape[0]))
for obs_index in obs_transfered:
    
    obs_operator[jjj, obs_index] = 1
    jjj+= 1
    
obs_operator = jax.numpy.asarray(obs_operator)  
Eigen = jax.numpy.asarray(Eigen)
Sigma = jax.numpy.asarray(Sigma)
pre_mat_stiff = jax.numpy.asarray(pre_mat_stiff_sparse.todense())

values = np.asarray([[pre_mat_stiff_sparse[i,j]] for i, j in zip(*pre_mat_stiff_sparse.nonzero())]).squeeze()
rows, cols = pre_mat_stiff_sparse.nonzero()

from functools import partial
partial(jax.jit, static_argnums=(4))
def sp_matmul(values, rows, cols, B, shape):
    """
    Arguments:
        A: (N, M) sparse matrix represented as a tuple (indexes, values)
        B: (M,K) dense matrix
        shape: value of N
    Returns:
        (N, K) dense matrix
    """
    
    # assert B.ndim == 2
    B = jnp.expand_dims(B, axis = 1)
    in_ = B.take(cols, axis=0)
    prod = in_*values[:, None]
    res = jax.ops.segment_sum(prod, rows, shape)
    
    return res

# Inverse-learned-network
def solve_forward(x):
    
    parameter_of_interest = jnp.dot(x, jnp.transpose(jnp.dot(Eigen, Sigma)))
    stiff = sp_matmul(values, rows, cols, jnp.exp(parameter_of_interest), dimension_of_PoI**2)
    stiff = jnp.reshape(stiff, (dimension_of_PoI, dimension_of_PoI))
    
    A = jnp.squeeze(jnp.take(stiff,free_index, axis=1))

    B = jnp.squeeze(jnp.take(A,free_index, axis=0))
    
    state = jnp.linalg.solve(B, load_vector)
    
    out_pred = jnp.dot(obs_operator, state).squeeze()
    
    return out_pred

def misfit(x,y, alpha):
    misfit = jnp.linalg.norm(solve_forward(x) - y)**2
    regulaziration = jnp.linalg.norm(x)**2
    
    return alpha * misfit + regulaziration

from jax._src.scipy.optimize.minimize import minimize as minimize

@jit
def solve_BFGS(y, alpha):
    x_0 = 0.01 * jax.random.normal(random.PRNGKey(case), single_test_parameter.shape)
    return (minimize(misfit, x_0, args = (y, alpha), method = 'BFGS', tol = 1e-16)).x

# # @jit
# def solve_BFGS(y, alpha):    
#     def f(x): 
#         return misfit(x,y, alpha)
    
#     solver = optax.lbfgs()
#     x_0 = 0.01 * jax.random.normal(random.PRNGKey(case), single_test_parameter.shape)
#     opt_state = solver.init(x_0)
#     value_and_grad = jax.jit(optax.value_and_grad_from_state(f))
    
#     for _ in range(100):
#         value, grad = value_and_grad(x_0, state=opt_state)
#         updates, opt_state = solver.update(grad, opt_state, x_0, value=value, grad=grad, value_fn=f)
#         x_0 = optax.apply_updates(x_0, updates)
#         # print('Objective function: ', f(x_0), np.linalg.norm(x_0))
#     return x_0
    
@jit
def acc(preds, true):
    return jnp.mean(jnp.square(preds - true)) / jnp.mean(jnp.square(true))

# Remove batch processing
# batched_acc = vmap(acc, in_axes = (0,0))

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
filename = f'Single_Sample_{SAMPLE_INDEX}_Alpha_Error_Results_case_{case}_Noise_01.txt'
header = "Alpha,Error"
np.savetxt(filename, results, fmt='%1.11e', delimiter=',', header=header, comments='')

# Save best result summary
summary_filename = f'Single_Sample_{SAMPLE_INDEX}_Best_Summary_case_{case}_Noise_01_inverse_solution'
np.save(summary_filename, best_inverted_params)