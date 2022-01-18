''' Libraries.'''
import numpy as np
import scipy
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution
from scipy.optimize import dual_annealing
#import nevergrad as ng


'''The module for training and execution of the Axon algorithm.'''
class axon_network():

    def __init__(self, input_dimension, activation_function):
        '''Inputs:
           "input_dimension"      ... dimension of the network input
                                      (= number of initial basis functions)
           "activation_function"  ... string determining which activation
                                      function is used
                                      (options: 'relu', 'leaky relu', 'repu',
                                      'tanh', 'sigmoid' or 'softplus')
           Output:
           The residual neural network and its attributes.'''

        self.in_dim = input_dimension
        self.activ_fct = activation_function
        self.k = 0


    '''The method for training the network using the Axon algorithm.'''
    def axon_algorithm(self, training_set, num_of_add_bases, optim_options,
                        stopping_criterion=None):
        for k in range(num_of_add_bases):
            early_stop = self.axon_step(training_set, optim_options,
                                        stopping_criterion)
            if early_stop:
                break
            self.k += 1
        # Compute the best approximation coefficients
        self.c_hat = np.dot(self.Q.T, training_set[-1,:].T)
        if early_stop:
            print("The training has stopped due to the stopping criterion.")


    '''The method for applying a single iteration of the Axon algorithm.'''
    def axon_step(self, training_set, optim_options, stopping_criterion):
        # Initialize the algorithm, if we haven't yet
        if self.k == 0:
            # Perform the reduced QR-decomposition of the training data inputs
            # and save it as R and Q
            self.Q,self.R = np.linalg.qr(training_set[:-1,:].T, mode='reduced')
            self.residuals = []
            self.residuals_length = []
            self.weights_hat = []

        # Compute the residual
        self.residuals.append(np.subtract(training_set[-1,:].T,
                     np.dot(self.Q, (np.dot(self.Q.T, training_set[-1,:].T)))))

        # Stop the learning if the stopping criterion is met (if provided)
        self.residuals_length.append(np.linalg.norm(self.residuals[-1]))
        if stopping_criterion is not None:
            if self.residuals_length[-1] < stopping_criterion:
                return True

        # Solve the minimization problem
        if optim_options[0] == 'basinhopping':
            # Randomly generate an initial guess for the maximizer
            x_start = np.random.rand(self.Q.shape[1])
            x_start = np.true_divide(x_start, np.linalg.norm(x_start))
            if optim_options[1] == 'classical':
                x_min = basinhopping(self.func_classical, x_start, niter=70,
                                     T=0.7)
            elif optim_options[1] == 'author':
                x_min = basinhopping(self.func_author, x_start, niter=70,
                                     T=0.7)
            elif optim_options[1] == 'opt':
                x_min = basinhopping(self.func_opt, x_start, niter=70,
                                     T=0.7)
        elif optim_options[0] == 'differential_evolution':
            # If we set updating='deferred' and workers=-1
            # all available CPU cores are used for the optimization.
            bounds = [(-1, 1)]
            bounds = bounds * self.Q.shape[1]
            if optim_options[1] == 'classical':
                x_min = differential_evolution(self.func_classical, bounds,
                                               maxiter=300, popsize=5,
                                               polish=False)
            elif optim_options[1] == 'author':
                x_min = differential_evolution(self.func_author, bounds,
                                               maxiter=300, popsize=5,
                                               polish=False)
            elif optim_options[1] == 'opt':
                x_min = differential_evolution(self.func_opt, bounds,
                                               maxiter=300, popsize=5,
                                               polish=False)
        elif optim_options[0] == 'dual_annealing':
            bounds = [(-1, 1)]
            bounds = bounds * self.Q.shape[1]
            if optim_options[1] == 'classical':
                x_min = dual_annealing(self.func_classical, bounds,
                                       maxiter=250)
            elif optim_options[1] == 'author':
                x_min = dual_annealing(self.func_author, bounds,
                                       maxiter=250)
            elif optim_options[1] == 'opt':
                x_min = dual_annealing(self.func_opt, bounds,
                                       maxiter=250)
        # elif optim_options[0] == 'one_plus_one':
        #     optimizer = ng.optimizers.OnePlusOne(
        #                           parametrization=self.Q.shape[1],
        #                           budget=5000)
        #     if optim_options[1] == 'classical':
        #         x_min = optimizer.minimize(lambda x: self.func_classical(x))
        #     elif optim_options[1] == 'author':
        #         x_min = optimizer.minimize(lambda x: self.func_author(x))
        #     elif optim_options[1] == 'opt':
        #         x_min = optimizer.minimize(lambda x: self.func_opt(x))
        else:
            raise ValueError("This optimization method is not supported.")
        if optim_options[0] == 'one_plus_one':
            self.weights_hat.append(x_min.args[0])
        else:
            self.weights_hat.append(x_min.x)

        # Compute the next basis phi
        phi = self.apply_activation(np.dot(self.Q, self.weights_hat[-1]))

        # Compute alpha and save it in R
        self.R = np.append(self.R, np.reshape(np.dot(self.Q.T, phi),
                                              (self.Q.shape[1],1)), axis=1)

        # Orthonormalize the basis using the Gram Schmidt algorithm
        for count in range(self.Q.shape[1]):
            phi = np.subtract(phi, self.Q[:,count]*self.R[count,-1])


        # Compute beta and save it in R
        self.R = np.append(self.R, np.zeros((1, self.R.shape[1])), axis=0)
        self.R[-1,-1] = np.linalg.norm(phi)

        # Normalize phi and add it to Q
        phi = np.reshape(np.true_divide(phi, self.R[-1,-1]), (phi.shape[0], 1))
        self.Q = np.append(self.Q, phi, axis=1)


    '''The methods for defining the objective functions.'''
    def func_classical(self, x):
        temp = self.apply_activation(np.dot(self.Q, x))
        temp1 = np.vdot(temp, temp)
        temp2 = np.absolute(np.dot(self.residuals[-1], temp))
        return -temp2/temp1

    def func_author(self, x):
        penalty_parameter = 1e-8
        temp = np.vdot(x, x)
        temp2 = np.dot(self.Q, x)
        if temp < penalty_parameter*10:
            return 100
        return -(np.dot(self.apply_activation(temp2),self.residuals[-1]))**2/ \
                (np.vdot(temp2, temp2)) + penalty_parameter*(temp-1)**2

    def func_opt(self, x):
        penalty_parameter = 1e-8
        temp = np.vdot(x, x)
        temp2 = np.dot(self.Q, x)
        if temp < penalty_parameter*10:
            return 100
        return -(np.dot(self.apply_activation(temp2),self.residuals[-1]))**2/ \
                temp + penalty_parameter*(temp-1)**2


    ''''The method for applying the activiation function.'''
    def apply_activation(self, inputs):
        if self.activ_fct == 'relu':
            return inputs.clip(min=0)
        elif self.activ_fct == 'leaky relu':
            alpha = 0.01
            factor = np.ones_like(inputs)
            factor[inputs < 0] = alpha
            return np.multiply(inputs, factor)
        elif self.activ_fct == 'repu':
            p = 2
            return np.power(inputs.clip(min=0), p)
        elif self.activ_fct == 'tanh':
            return np.tanh(inputs)
        elif self.activ_fct == 'sigmoid':
            return np.true_divide(1, 1+np.exp(-inputs))
        elif self.activ_fct == 'softplus':
            return np.log(1 + np.exp(inputs))
        else:
            raise ValueError("The 'activation_function' has to be either "
                             "'relu', 'leaky relu', 'repu', 'tanh', 'sigmoid' "
                             "or 'softplus'.")


    '''The method for measuring the performance of the network on the
       validation set.'''
    def validate(self, validation_set, error_measure):
        diff = np.subtract(validation_set[-1,:],
                           self.forward_propagation(validation_set[:-1,:]))
        # Compute the mean square error
        if error_measure == 'MSE':
            error_val = np.vdot(diff, diff)
            if validation_set.shape[1] > 1:
                error_val = (1/validation_set.shape[1]) * error_val
            # print("The mean square error is: ", error_val)
            return error_val
        # Compute the worst case error
        elif error_measure == 'WCE':
            error_val = np.linalg.norm(diff, ord=np.inf)
            # print("The worst case error is: ", error_val)
            return error_val
        # Compute the maximal relative error
        elif error_measure == 'max RE':
            error_val = np.true_divide(diff, validation_set[-1,:])
            error_val = np.absolute(error_val)
            error_val = max(error_val)
            # print("The maximal relative error is: ", error_val)
            return error_val
        else:
            raise ValueError("The 'error_measure' has to be either "
                             "'MSE', 'WCE' or 'max RE'.")


    '''The method for computing the network output for some input x.'''
    def forward_propagation(self, x, num_neurons):
        x = scipy.linalg.solve_triangular(self.R[:self.in_dim,:self.in_dim].T,
                                          x, lower=True, overwrite_b=False)
        for a in range(num_neurons):
            x_new = self.apply_activation(np.dot(self.weights_hat[a], x))
            x_new = np.subtract(x_new,
                             np.dot(self.R[:self.in_dim+a,self.in_dim+a].T, x))
            x_new = np.true_divide(x_new, self.R[self.in_dim+a,self.in_dim+a])
            x_new = np.reshape(x_new, (1, x_new.shape[0]))
            x = np.append(x, x_new, axis=0)
        return np.dot(self.c_hat[:x.shape[0]], x)


    '''The method for computing the derivative of the network w.r.t. the input.'''
    def compute_gradient(self, x, num_neurons):
        v_inter = np.full((40,x.shape[1]), 1.0)
        x = scipy.linalg.solve_triangular(self.R[:self.in_dim,:self.in_dim].T,
                                          x, lower=True, overwrite_b=False)
        for a in range(num_neurons):
            v_inter[a,:] = np.dot(self.weights_hat[a], x)
            x_new = self.apply_activation(v_inter[a,:])
            
            x_new = np.subtract(x_new,
                             np.dot(self.R[:self.in_dim+a,self.in_dim+a].T, x))
            x_new = np.true_divide(x_new, self.R[self.in_dim+a,self.in_dim+a])
            x_new = np.reshape(x_new, (1, x_new.shape[0]))
            x = np.append(x, x_new, axis=0)
        
        # Forward accumulation dx and dy
        v_dx = np.full((43,x.shape[1]), 1.0)
        v_inter_dx = np.full((40,x.shape[1]), 1.0)
        v_tilde_dx = np.full((43,x.shape[1]), 1.0)
        v_dx[1,:] = 0 # to differentiate w.r.t. 'x'
        v_dx[2,:] = 0
        v_tilde_dx[0,:] = np.true_divide(v_dx[0,:], self.R[0,0])
        v_tilde_dx[1,:] = np.true_divide(v_dx[1,:]-self.R[0,1]*v_tilde_dx[0,:],
                                         self.R[1,1])
        v_tilde_dx[2,:] = np.true_divide(v_dx[2,:]-self.R[0,2]*v_tilde_dx[0,:]-self.R[1,2]*v_tilde_dx[1,:],
                                         self.R[2,2])
        
        v_dy = np.full((43,x.shape[1]), 1.0)
        v_inter_dy = np.full((40,x.shape[1]), 1.0)
        v_tilde_dy = np.full((43,x.shape[1]), 1.0)
        v_dy[0,:] = 0 # to differentiate w.r.t. 'y'
        v_dy[2,:] = 0
        v_tilde_dy[0,:] = 0
        v_tilde_dy[1,:] = np.true_divide(v_dy[1,:], self.R[1,1])
        v_tilde_dy[2,:] = np.true_divide(v_dy[2,:]-self.R[1,2]*v_tilde_dy[1,:],
                                         self.R[2,2])
        for index in range(num_neurons):
            weights_net = self.weights_hat[index]
            v_inter_dx[index,:] = np.dot(weights_net, v_tilde_dx[:index+3,:])
            v_dx[index+3,:] = self.apply_activation_grad(v_inter[index,:]) * v_inter_dx[index,:]
            v_tilde_dx[index+3,:] = np.true_divide(v_dx[index+3,:]-np.dot(self.R[:index+3,index+3],v_tilde_dx[:index+3,:]),
                                                   self.R[index+3,index+3])
            
            v_inter_dy[index,:] = np.dot(weights_net, v_tilde_dy[:index+3,:])
            v_dy[index+3,:] = self.apply_activation_grad(v_inter[index,:]) * v_inter_dy[index,:]
            v_tilde_dy[index+3,:] = np.true_divide(v_dy[index+3,:]-np.dot(self.R[:index+3,index+3],v_tilde_dy[:index+3,:]),
                                                   self.R[index+3,index+3])
        if num_neurons == 0:
            dx = np.dot(self.c_hat[:self.in_dim], v_tilde_dx[:self.in_dim,:])
            dy = np.dot(self.c_hat[:self.in_dim], v_tilde_dy[:self.in_dim,:])
        else:
            index += 1
            dx = np.dot(self.c_hat[:self.in_dim+index], v_tilde_dx[:self.in_dim+index,:])
            dy = np.dot(self.c_hat[:self.in_dim+index], v_tilde_dy[:self.in_dim+index,:])
        return np.array([dx, dy])


    def forward_pass(self, x):
        v = np.full((43,x.shape[1]), 1.0)
        v[:3,:] = x[:3,:]
        x = scipy.linalg.solve_triangular(self.R[:self.in_dim,:self.in_dim].T,
                                          x, lower=True, overwrite_b=False)
        for a in range(self.k):
            x_new = self.apply_activation(np.dot(self.weights_hat[a], x))
            v[3+a,:] = x_new
            x_new = np.subtract(x_new,
                             np.dot(self.R[:self.in_dim+a,self.in_dim+a].T, x))
            x_new = np.true_divide(x_new, self.R[self.in_dim+a,self.in_dim+a])
            x_new = np.reshape(x_new, (1, x_new.shape[0]))
            x = np.append(x, x_new, axis=0)
        return v
        # v = np.full((43,), 1.0)
        # v_inter = np.full((40,), 1.0)
        # v_tilde = np.full((43,), 1.0)
        # # Initial basis functions
        # v[0] = x
        # v[1] = y
        # #v[2] = 1.0
        # v_tilde[0] = v[0] / self.R[0,0]
        # v_tilde[1] = (v[1]-self.R[0,1]*v_tilde[0]) / self.R[1,1]
        # v_tilde[2] = (v[2]-self.R[0,2]*v_tilde[0]-self.R[1,2]*v_tilde[1]) \
        #              / self.R[2,2]
        # # Additional basis functions
        # for index,weights_net in enumerate(self.weights_hat):
        #     v_inter[index] = np.dot(weights_net, v_tilde[:index+3])
        #     v[index+3] = self.apply_activation(v_inter[index])
        #     v_tilde[index+3] = (v[index+3] - np.dot(self.R[:index+3,index+3], v_tilde[:index+3])) \
        #                        / self.R[index+3,index+3]
        # f_NN = np.dot(self.c_hat, v_tilde)
        # return [v, v_inter, v_tilde, f_NN]




    def apply_activation_grad(self, inputs):
        if self.activ_fct == 'relu':
            output = np.ones_like(inputs, dtype=float)
            output[np.argwhere(inputs<0)] = 0.0
            return output
        elif self.activ_fct == 'leaky relu':
            alpha = 0.01
            output = np.ones_like(inputs, dtype=float)
            output[np.argwhere(inputs<0)] = alpha
            return output
        elif self.activ_fct == 'repu':
            #p = 2
            output = inputs.clip(min=0)
            return output*2
        elif self.activ_fct == 'tanh':
            return None
        elif self.activ_fct == 'sigmoid':
            return None
        elif self.activ_fct == 'softplus':
            return None
        else:
            raise ValueError("The 'activation_function' has to be either "
                             "'relu', 'leaky relu', 'repu', 'tanh', 'sigmoid' "
                             "or 'softplus'.")


