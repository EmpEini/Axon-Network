'''Libraries & Modules:'''
import generate_data
import axon_resnet
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from timeit import default_timer as timer

'''------------------------------------------------------------------------'''
'''-------------------------Start of specifications------------------------'''
'''------------------------------------------------------------------------'''
'''Dataset specifications'''
# example_number: 1D [0,1,2,3]
#                 2D [4,5,6,7,8,9] (use square dataset sizes)
#                 3D []  (use cubic dataset sizes)
example_number = 1
train_set_size = 10000 # size of training set
valid_set_size = 100000 # size of validation set
# From which distribution are the training nodes drawn from?
# training_data_distribution: 'uniform', 'chebyshev' or 'equidistant'
training_data_distribution = 'chebyshev'
lower_bounds = [0] # lower bound for each dimension
upper_bounds = [1] # upper bound for each dimension
# Specifies the initial basis functions. This is the network input.
# initial_bases: 'identity' or specify on your on own in generate_data.py
initial_bases = 'identity'
# Should the data to be noisy?
# noise: None or 'Gaussian'
noise = None

'''Neural network specifications'''
# Network specifications
# activation_function: 'relu', 'leaky relu', 'repu', 'tanh',
#                      'sigmoid' or 'softplus'
activation_function = 'relu'

'''Training specifications'''
# algorithm: 'basinhopping', 'differential_evolution', 'dual_annealing'
#             or 'one_plus_one'
algorithm = 'one_plus_one'
# obj_fct: 'classical', 'FokinaOseledets' or 'opt'
obj_fct = 'opt'

'''Validation specifications'''
# error_measure: 'MSE', 'WCE' or 'max RE'
error_measure = 'MSE'

'''Runtime specifications'''
num_runs = 20 # How often do you want to run the program?
# How many iterations of the Axon algorithm do you want to perform?
max_num_add_bases = 40
# To avoid overfitting you can add a termination threshold.
# If provided, the learning process terminates when the current residual
# length falls below the threshold.
# threshold: None or a floating point number
threshold = None

'''Plot specifications'''
# Do you want to plot the training & validation nodes?
# dataset_plot: None or 'hexbin' (None <=> no plot)
dataset_plot = 'hexbin'
time_plot = True # Do you want to plot the time spent in training?
plot_approx_fct = True # Do you want to plot the fct. vs the NN fct. approx.?
plot_basis_fcts = True # Do you want to plot the first few basis fcts?
num_of_basis_fcts = 6 #  How many basis function do you want to plot?

'''Additional output specifications'''
# Do you want to print the number of the best run regarding the error measure?
print_best = True
check_orthogonality = True # Do want to check if we have lost orthogonality?
'''------------------------------------------------------------------------'''
'''--------------------------End of specifications-------------------------'''
'''------------------------------------------------------------------------'''

# Generate the training set
training_class = generate_data.data_generator(example_number, train_set_size,
                                              training_data_distribution,
                                              lower_bounds, upper_bounds,
                                              initial_bases, noise)
training_set = training_class.dataset

# Generate the validation set
validation_class = generate_data.data_generator(example_number, valid_set_size,
                                                'uniform', lower_bounds,
                                                upper_bounds, initial_bases)
validation_set = validation_class.dataset

# Fix the neural network input dimension for all runs
input_dimension = validation_set.shape[0] - 1

# Array allocation for plots
x_axis = np.arange(max_num_add_bases+1)
y_error = np.zeros((num_runs, max_num_add_bases+1), dtype=float)
save_network_param = [None] * num_runs
y_time = np.zeros((num_runs, max_num_add_bases+1), dtype=float)

'''Create, train and validate the network multiple times'''
for l in range(num_runs):
    # Iteration 0 - validation
    print('Run: ', l+1, '// Iteration: ', 0)
    net = axon_resnet.axon_network(input_dimension, activation_function)
    net.Q, net.R = np.linalg.qr(training_set[:-1,:].T, mode='reduced')
    net.c_hat = np.dot(net.Q.T, training_set[-1,:].T)
    y_error[l,0] = net.validate(validation_set, error_measure)
    print(error_measure + ': ' + str(y_error[l,0]))
    for i in range(1, max_num_add_bases+1):
        print('Run: ', l+1, '// Iteration: ', i)
        start = timer()
        net.axon_algorithm(training_set, 1, [algorithm, obj_fct], threshold)
        end = timer()
        y_error[l,i] = net.validate(validation_set, error_measure)
        print(error_measure + ': ' + str(y_error[l,i]))
        y_time[l,i] = y_time[l,i] + end - start
    save_network_param[l] = net

# Determine which network performed the best over 'num_runs' many runs
best_perf_run_idx = 0
for i in range(1,num_runs):
    if (np.min(y_error[best_perf_run_idx,:]) > np.min(y_error[i,:])):
        best_perf_run_idx = i
net = save_network_param[best_perf_run_idx]
if print_best:
    print('The best run in terms of the ' + error_measure + ' was run ' +
          str(best_perf_run_idx+1) + '.')

# Check the semi-orthogonality of Q
if check_orthogonality:
    for i in range(num_runs):
        network = save_network_param[i]
        unit_approx = np.dot(network.Q.T, network.Q)
        if np.allclose(unit_approx, np.eye(unit_approx.shape[0])):
            pass
        else:
            print('Run ' + str(i+1) + ': Semi-orthogonality is lost!')

'''Plot routine'''
# Adjust the xlim, ylim, and zlim for the plots
if plot_approx_fct or plot_basis_fcts:
    plot_temp1 = np.absolute(upper_bounds[0]-lower_bounds[0])
    plot_temp1 = plot_temp1 * 0.08
    xlim_bounds = [lower_bounds[0]-plot_temp1, upper_bounds[0]+plot_temp1]
    if input_dimension > 2:
        plot_temp2 = np.absolute(upper_bounds[1]-lower_bounds[1])
        plot_temp2 = plot_temp2 * 0.08
        ylim_bounds = [lower_bounds[1]-plot_temp2, upper_bounds[1]+plot_temp2]
    if input_dimension > 3:
        plot_temp3 = np.absolute(upper_bounds[2]-lower_bounds[2])
        plot_temp3 = plot_temp3 * 0.08
        zlim_bounds = [lower_bounds[2]-plot_temp3, upper_bounds[2]+plot_temp3]

# Plot the training- and the validation set
if dataset_plot is not None and input_dimension in [2, 3]:
    if input_dimension == 2:
        fig1, ax1 = plt.subplots(figsize=plt.figaspect(0.58))
        ax1.plot(validation_set[0,:], validation_set[-1,:], linewidth=2,
                 color='C1', zorder=1)
        ax1.scatter(training_set[0,:], training_set[-1,:], s=2, c='black',
                    zorder=2)
        ax1.set_title('Training set vs. validation set')
        ax1.set_xlim(xlim_bounds)
    elif input_dimension == 3:
        if dataset_plot == 'hexbin':
            fig1, ax1 = plt.subplots(figsize=plt.figaspect(0.58))
            h1 = ax1.hexbin(validation_set[0,:], validation_set[1,:],
                            validation_set[-1,:], cmap=cm.gist_earth,
                            marginals=True, zorder=1)
            fig1.colorbar(h1, shrink=0.8, aspect=18)
            ax1.scatter(training_set[0,:], training_set[1,:], s=3, c='black',
                        zorder=2)
            ax1.set_title('Training set vs. validation set')
            ax1.set_xlim(xlim_bounds)
            ax1.set_ylim(ylim_bounds)
        else:
            raise ValueError("The 'dataset_plot' has to be 'hexbin' or None.")

# Error plot
fig2, ax2 = plt.subplots(figsize=plt.figaspect(0.58))
ax2.set_yscale('log')
ax2.set_title(str(error_measure))
ax2.plot(x_axis, y_error[best_perf_run_idx,:], linewidth=2, color='C3')

# Time plot
if time_plot:
    fig3, ax3 = plt.subplots(figsize=plt.figaspect(0.58))
    ax3.set_title('Training time in each iteration')
    legend_list = []
    for i in range(num_runs):
        ax3.plot(x_axis, y_time[i,:], linewidth=2)
        legend_list.append('Run ' + str(i+1))
    if num_runs > 1:
        ax3.legend(legend_list)

# Function approximation plot
if plot_approx_fct:
    if input_dimension == 2:
        num = 1000
    if input_dimension == 3:
        num = 6400
    plot_data = generate_data.data_generator(example_number, num,
                                             'equidistant', lower_bounds,
                                             upper_bounds, initial_bases)
    plot_data = plot_data.dataset
    if input_dimension == 2:
        f_NN_x = net.forward_propagation(plot_data[:-1,:])
        fig4, ax4 = plt.subplots(figsize=plt.figaspect(0.58))
        ax4.plot(plot_data[0,:], plot_data[-1,:], linewidth=3, color='C0',
                  zorder=1)
        ax4.plot(plot_data[0,:], f_NN_x, linewidth=1.8, color='C8', zorder=2)
        ax4.set_title('Function output vs. neural network output')
        ax4.legend(['function', 'neural network'])
        ax4.set_xlim(xlim_bounds)
        # Difference plot
        fig5, ax5 = plt.subplots(figsize=plt.figaspect(0.58))
        f_diff = plot_data[-1,:] - f_NN_x
        ax5.scatter(plot_data[0,:], f_diff, s=0.02, c='red')
        ax5.set_title('Difference between function outputs and neural network '
                      'outputs')
        ax5.set_xlim(xlim_bounds)
    elif input_dimension == 3:
        # Difference plot
        f_NN_x = net.forward_propagation(plot_data[:-1,:])
        f_diff = plot_data[-1,:] - f_NN_x
        fig4, ax4 = plt.subplots(figsize=plt.figaspect(0.58))
        h4 = ax4.hexbin(plot_data[0,:], plot_data[1,:], f_diff,
                        cmap=cm.gist_earth, marginals=True)
        fig4.colorbar(h4, shrink=0.8, aspect=18)
        ax4.set_title('Difference between function outputs and neural network '
                      'outputs')
        ax4.set_xlim(xlim_bounds)
        ax4.set_ylim(ylim_bounds)

# Basis function plot
if plot_basis_fcts:
    base_plot = training_set[:-2,:]
    orth_base_plot = training_set[:-2,:]
    for s in range(num_of_basis_fcts):
            base_plot = np.vstack((base_plot, net.apply_activation(np.dot(
                            net.Q[:,:input_dimension+s], net.weights_hat[s]))))
            orth_base_plot = np.vstack((orth_base_plot,
                                        net.Q[:,input_dimension+s]))
    figure_list = [None] * num_of_basis_fcts * 2
    axes_list = [None] * num_of_basis_fcts * 2
    if input_dimension == 2:
        for s in range(num_of_basis_fcts):
            figure_list[s], axes_list[s] = plt.subplots(
                                                   figsize=plt.figaspect(0.58))
            axes_list[s].plot(base_plot[0,:], base_plot[1+s,:], linewidth=2,
                              color='C4')
            axes_list[s].set_title('Basis function ' + str(s+1))
            axes_list[s].set_xlim(xlim_bounds)
        for r in range(num_of_basis_fcts):
            figure_list[s+r+1], axes_list[s+r+1] = plt.subplots(
                                                   figsize=plt.figaspect(0.58))
            axes_list[s+r+1].plot(orth_base_plot[0,:], orth_base_plot[1+r,:],
                                  linewidth=2, color='C5')
            axes_list[s+r+1].set_title('Orthogonal basis function ' + str(r+1))
            axes_list[s+r+1].set_xlim(xlim_bounds)
    elif input_dimension == 3:
        hexbin_list = [None] * num_of_basis_fcts * 2
        for s in range(num_of_basis_fcts):
            figure_list[s], axes_list[s] = plt.subplots(
                                                   figsize=plt.figaspect(0.58))
            hexbin_list[s] = axes_list[s].hexbin(base_plot[0,:],
                                base_plot[1,:], base_plot[2+s,:],
                                cmap=cm.gist_earth, marginals=True)
            figure_list[s].colorbar(hexbin_list[s], shrink=0.8, aspect=18)
            axes_list[s].set_title('Basis function ' + str(s+1))
            axes_list[s].set_xlim(xlim_bounds)
            axes_list[s].set_ylim(ylim_bounds)
        for r in range(num_of_basis_fcts):
            figure_list[s+r+1], axes_list[s+r+1] = plt.subplots(
                                                   figsize=plt.figaspect(0.58))
            hexbin_list[s+r+1] = axes_list[s+r+1].hexbin(orth_base_plot[0,:],
                                    orth_base_plot[1,:], orth_base_plot[2+r,:],
                                    cmap=cm.gist_earth, marginals=True)
            figure_list[s+r+1].colorbar(hexbin_list[s+r+1], shrink=0.8,
                                        aspect=18)
            axes_list[s+r+1].set_title('Orthogonal basis function ' + str(r+1))
            axes_list[s+r+1].set_xlim(xlim_bounds)
            axes_list[s+r+1].set_ylim(ylim_bounds)


