''' Libraries:'''
import numpy as np
from scipy import special

'''The module for generating datasets.'''
class data_generator():
    def __init__(self, example_number, dataset_size, distribution,
                 lower_bounds, upper_bounds, inital_bases, noise=None):
        '''Inputs:
           "example_number"   ... integer determining which function to approx.
           "dataset_size"     ... size of the dataset
           "distribution"     ... string - distribution of the data
                                  (options: 'uniform', 'chebyshev' or
                                            'equidistant')
           "lower_bounds"     ... list containing the lower dim. boundaries, 
                                  (i.e. xy_lower_bound  = [0,-1]
                                   corresponds to 0 <= x, -1 <= y)
           "uppper_bounds"    ... list containing the upper dim. boundaries
           "inital_bases"     ... string - specifies the inital basis functions
                                  (options: 'identity')
           "noise"            ... string - add noise to the data (additive)
                                  (options: None or 'Gaussian')
           Output:
           The dataset and its attributes.'''

        self.example_number = example_number
        self.dataset_size = dataset_size
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.inital_bases = inital_bases
        self.noise = noise
        # If there is no noise, set the noise vector to 0
        self.noise_add = np.zeros((self.dataset_size,))

        # Determine the domain dimension given the Example number
        if self.example_number in [0,1,2,3]: # 1-dimensional input
            self.domain_dimension = 1
        elif self.example_number in [4,5,6,7,8,9]: # 2-dimensional input
            self.domain_dimension = 2
        elif self.example_number in []: # 3-dimensional input
            self.domain_dimension = 3
        else:
            raise ValueError("There is no example number " +
                             str(example_number))

        # Allocation of the data array
        self.dataset = np.empty((self.domain_dimension+2, self.dataset_size),
                                dtype=float)

        # The first 'domain_dimension' many rows contain the provided basis
        # function outputs for the dataset.
        if distribution == 'uniform':
            # Draw 'dataset_size' many samples from the univariate uniform
            # distributions on [0,1) and adjust it according to the boundaries
            self.dataset[:-2,:] = np.random.uniform(self.lower_bounds,
                                                    self.upper_bounds,
                                                    (self.dataset_size,
                                                     self.domain_dimension)).T
        elif distribution == 'chebyshev':
            if self.domain_dimension == 1:
                data = (2 * np.arange(1,self.dataset_size+1)) - 1
                data = np.cos((np.pi/(2*self.dataset_size)) * data)
                # Translate the chebyshev nodes onto the interval [lb,ub]
                data = 0.5*(self.lower_bounds[0]+self.upper_bounds[0]) + \
                       0.5*(self.upper_bounds[0]-self.lower_bounds[0]) * data
            elif self.domain_dimension == 2:
                # To generate 2D chebyshev nodes take square numbers
                # as dataset_size
                square_root = int(round(self.dataset_size**(1./2)))
                data_temp = (2 * np.arange(1,square_root+1)) - 1
                data_temp = np.cos((np.pi/(2*square_root)) * data_temp)
                # Translate the chebyshev nodes onto the interval [lb[0],ub[0]]
                data_1 = 0.5*(self.lower_bounds[0]+self.upper_bounds[0]) + \
                         0.5*(self.upper_bounds[0]-self.lower_bounds[0]) * \
                         data_temp
                # Translate the chebyshev nodes onto the interval [lb[1],ub[1]]
                data_2 = 0.5*(self.lower_bounds[1]+self.upper_bounds[1]) + \
                         0.5*(self.upper_bounds[1]-self.lower_bounds[1]) * \
                         data_temp
                # Generate the cartesian product of data_1 and data_2
                data = np.array(np.meshgrid(data_1, data_2)).T.reshape(-1,2)
                data = data.T
            elif self.domain_dimension == 3:
                # To generate 3D chebyshev nodes take cubic numbers
                # as dataset_size
                third_root = int(round(self.dataset_size**(1./3)))
                data_temp = (2 * np.arange(1,third_root+1)) - 1
                data_temp = np.cos((np.pi/(2*third_root)) * data_temp)
                data_1 = 0.5*(self.lower_bounds[0]+self.upper_bounds[0]) + \
                         0.5*(self.upper_bounds[0]-self.lower_bounds[0]) * \
                         data_temp
                data_2 = 0.5*(self.lower_bounds[1]+self.upper_bounds[1]) + \
                         0.5*(self.upper_bounds[1]-self.lower_bounds[1]) * \
                         data_temp
                data_3 = 0.5*(self.lower_bounds[2]+self.upper_bounds[2]) + \
                         0.5*(self.upper_bounds[2]-self.lower_bounds[2]) * \
                         data_temp
                data = np.array(np.meshgrid(data_1, data_2,
                                            data_3)).T.reshape(-1,3)
                data = data.T
            self.dataset[:-2,:] = data
        elif distribution == 'equidistant':
            if self.domain_dimension == 1:
                data = np.linspace(self.lower_bounds[0], self.upper_bounds[0],
                                   self.dataset_size)
            elif self.domain_dimension == 2:
                square_root = int(round(self.dataset_size**(1./2)))
                data_1 = np.linspace(self.lower_bounds[0],
                                     self.upper_bounds[0], square_root)
                data_2 = np.linspace(self.lower_bounds[1],
                                     self.upper_bounds[1], square_root)
                # Generate the cartesian product of data_1 and data_2
                data = np.array(np.meshgrid(data_1, data_2)).T.reshape(-1,2)
                data = data.T
            elif self.domain_dimension == 3:
                third_root = int(round(self.dataset_size**(1./3)))
                data_1 = np.linspace(self.lower_bounds[0],
                                     self.upper_bounds[0], third_root)
                data_2 = np.linspace(self.lower_bounds[1],
                                     self.upper_bounds[1], third_root)
                data_3 = np.linspace(self.lower_bounds[2],
                                     self.upper_bounds[2], third_root)
                data = np.array(np.meshgrid(data_1, data_2,
                                            data_3)).T.reshape(-1,3)
                data = data.T
            self.dataset[:-2,:] = data
        else:
            raise ValueError("The 'distrubtion' has to be either 'uniform', "
                             "'chebyshev' or 'equidistant'")

        # Sort the data in ascending order according to the first row,
        # and break ties according to the subsequent rows
        if self.domain_dimension == 1:
            self.dataset = np.sort(self.dataset)
        elif self.domain_dimension == 2:
            self.dataset = self.dataset[:, np.lexsort((self.dataset[1,:],
                                                       self.dataset[0,:]))]
        elif self.domain_dimension == 3:
            self.dataset = self.dataset[:, np.lexsort((self.dataset[2,:],
                                                       self.dataset[1,:],
                                                       self.dataset[0,:]))]

        # The last component of the input data for the initial basis
        # functions is the constant function f(x)=1
        # (In traditional networks this is the bias)
        self.dataset[-2,:] = np.ones((self.dataset_size,))
        
        # Compute the gradient of the solution function at the datapoints
        self.gradient_f = self.compute_gradient(self.dataset[:-2,:])
        
        # Compute the outputs of the initial basis functions from (x,1)
        # (If needed, specify your own initial basis functions here)
        if self.inital_bases == 'identity':
            pass
        elif self.inital_bases == 'periodic':
            # first row is the sine function applied elementwise
            self.dataset[0,:] = np.sin(self.dataset[0,:])
            # second row is the cosine function applied elementwise
            self.dataset[1,:] = np.cos(self.dataset[1,:])
        else:
            raise ValueError("The 'initial_bases' have yet to be specified.")

        # The last row contains the function outputs
        self.dataset[-1,:] = self.apply_function(self.dataset[:-2,:])
        if self.noise is not None: # If noise=True, add noise to the data
            if self.noise == 'Gaussian':
                mean = 0
                std_dev = 0.2
                self.noise_add = np.random.normal(mean, std_dev,
                                                self.dataset.shape[1])
                self.dataset[-1,:] = self.dataset[-1,:] + self.noise_add
            else:
                raise ValueError("The 'noise' has to be either None or "
                                 "'Gaussian'")
        

    def apply_function(self, data):
        if self.example_number == 0: # ODE (Chapter 4)
            # -0.01*u''(x) + u(x) = 1  for x in (0,1)
            # u(0) = u(1) = 0
            # solution: u(x) = (1-exp(100))/(exp(200)-1)*exp(100x) +
            #                  (exp(100)-exp(200))/(exp(200)-1)*exp(-100x)+1
            a = np.exp(200)-1
            b = (1-np.exp(100))/a
            c = (np.exp(100)-np.exp(200))/a
            data_f = b*np.exp(100*data)+c*np.exp(-100*data)+1
        elif self.example_number == 1: # Logistic differential equation
            # u'(x) = u(x) - u^2(x)  for x in (0,1)
            # u(0) = 1/2
            # solution: u(x) = exp(x)/(exp(x)+1)
            a = np.exp(data)
            data_f = np.true_divide(a, a+1)
        elif self.example_number == 2: # Poisson equation in 1D
            # u''(x) = cos(2*pi*x)  for x in (0,1)
            # u(0) = u(1) = 0
            # solution: u(x) = sin^2(pi*x)/2(pi^2)
            a = np.square(np.sin(data*np.pi))
            b = 2*(np.pi)**2
            data_f = np.true_divide(a, b)
        elif self.example_number == 3: # Legendre's differential equation
            # (1-x^2)u''(x) - 2xu'(x) + 2u(x) = 0  for x in (-1/2,1/2)
            # u(-1/2) = u(1/2) = 2 - 1/2*log(3)
            # solution: u(x) = 2 + x*log((1-x)/(1+x))
            a = np.log(np.true_divide(1-data, 1+data))
            data_f = np.multiply(data, a) + 2
        elif self.example_number == 4: # Laplace equation in 2D
            # -u_xx(x,y) - u_yy(x,y) = 0  for (x,y) in (0,pi)^2
            # u(x,pi) = sin(x)            for x in [0,pi]
            # u(x,0) = 0                  for x in [0,pi]
            # u(0,y) = u(pi,y) = 0        for y in (0,pi)
            # solution: u(x,y) = sin(x)*(exp(y)-exp(-y))/(exp(pi)-exp(-pi))
            a = np.multiply(np.sin(data[0]), np.exp(data[1])-np.exp(-data[1]))
            b = np.exp(np.pi)-np.exp(-np.pi)
            data_f = np.true_divide(a, b)
        elif self.example_number == 5: # Helmholtz equation
            # u_xx(x,y) + u_yy(x,y) + u(x,y) = 0  for (x,y) in (-1,1)^2
            # u_n(x,-1) = u_n(x,1) = 1            for x in [-1,1]
            # u_n(-1,y) = u_n(1,y) = 1            for y in (-1,1)
            # solution: u(x,y) = -(cos(1+x)+cos(1-x)+cos(1+y)+cos(1-y))/sin(2)
            a = np.cos(1+data[0]) + np.cos(1-data[0])
            b = np.cos(1+data[1]) + np.cos(1-data[1])
            data_f = np.true_divide(a+b, -np.sin(2))
        elif self.example_number == 6: # Heat equation in 1D
            # u_t(x,t) = u_xx(x,t)  for (x,t) in (0,pi)x(0,1)
            # u(0,t) = u(pi,t) = 0  for t in [0,1)
            # u(x,0) = sin(x)       for x in [0,pi)
            # solution: u(x,t) = sin(x)*exp(-t)
            data_f = np.multiply(np.sin(data[0]), np.exp(-data[1]))
        elif self.example_number == 7: # Inhom. heat equation in 1D
            # u_t(x,t) = u_xx(x,t) + x  for (x,t) in (-1,1)x(0,1)
            # u(x,0) = 2                for x in [-1,0]
            # u(x,0) = 0                for x in (0,1]
            # solution: u(x,t) = 1+erf(x/sqrt(4t))+xt
            a = np.true_divide(data[0], 2*np.sqrt(data[1]))
            b = special.erf(a)
            c = np.multiply(data[0], data[1])
            data_f = 1 + b + c
        elif self.example_number == 8: # Wave equation in 1D
            # u_tt(x,t) - u_xx(x,t) = 0  for (x,t) in (-pi,pi)x(0,3)
            # u(x,0) = cos(x)            for x in [-pi,pi]
            # u_t(x,0) = 4cos(x)^2
            # solution: u(x,t) = cos(x)cos(t)+cos(2x)sin(2t)+2t
            a = np.multiply(np.cos(data[0]), np.cos(data[1]))
            b = np.multiply(np.cos(2*data[0]), np.sin(2*data[1]))
            data_f = a + b + 2*data[1]
        elif self.example_number == 9: # Inhom. wave equation in 1D
            # u_tt(x,t) - u_xx(x,t) = 3exp(x-2t)  for (x,t) in (-1,1)x(0,1)
            # u(x,0) = exp(x)                     for x in [-1,1]
            # u_t(x,0) = -2exp(x)                 for x in [-1,1]
            # solution: u(x,t) = exp(x-2t)
            data_f = np.exp(data[0]-2*data[1])
        return data_f
    
    
    def compute_gradient(self, data):
        if self.example_number == 0: # ODE (Chapter 4)
            # No implementation so far
            data_grad_f = data
        elif self.example_number == 1: # Logistic differential equation
            # Derivative: u'(x) = exp(x)/(exp(x)+1)^2
            a = np.square(1 + np.exp(data))
            data_grad_f = np.true_divide(np.exp(data), a)
        elif self.example_number == 2: # Poisson equation in 1D
            # Derivative: u'(x) = sin(pi*x)*cos(pi*x)/pi
            a = np.pi*data
            b = np.sin(a)
            c = np.cos(a)
            data_grad_f = np.true_divide(b*c, np.pi)
        elif self.example_number == 3: # Legendre's differential equation
            # Derivative: u'(x) = log((1-x)/(1+x)) + -2x/((1-x)(1+x))
            a = 1-data
            b = 1+data
            c = np.log(np.true_divide(a, b))
            d = np.true_divide(-2*data, a*b)
            data_grad_f = c+d
        elif self.example_number == 4: # Laplace equation in 2D
            # Derivatives: du(x,y)/dx = cos(x)*(exp(y)-exp(-y))/(exp(pi)-exp(-pi))
            #              du(x,y)/dy = sin(x)*(exp(y)+exp(-y))/(exp(pi)-exp(-pi))
            a = np.exp(np.pi)-np.exp(-np.pi)
            b = np.cos(data[0]) * (np.exp(data[1])-np.exp(-data[1]))
            partial_x = np.true_divide(b, a)
            c = np.sin(data[0]) * (np.exp(data[1])+np.exp(-data[1]))
            partial_y = np.true_divide(c, a)
            data_grad_f = np.vstack((partial_x,partial_y))
        elif self.example_number == 5: # Helmholtz equation
            # Derivatives: du(x,y)/dx = (sin(1+x)-sin(1-x))/sin(2)
            #              du(x,y)/dy = (cos(1+x)-cos(1-x))/sin(2)
            a = np.sin(2)
            partial_x = np.true_divide(np.sin(1+data[0])-np.sin(1-data[0]), a)
            partial_y = np.true_divide(np.sin(1+data[1])-np.sin(1-data[1]), a)
            data_grad_f = np.vstack((partial_x,partial_y))
        elif self.example_number == 6: # Heat equation in 1D
            # Derivatives: du(x,t)/dx = cos(x)*exp(-t)
            #              du(x,t)/dt = -sin(x)*exp(-t)
            partial_x = np.cos(data[0]) * np.exp(-data[1])
            partial_y = np.sin(data[0]) * np.exp(-data[1]) * (-1)
            data_grad_f = np.vstack((partial_x,partial_y))
        elif self.example_number == 7: # Inhom. heat equation in 1D
            # Derivatives: du(x,t)/dx = ...
            #              du(x,t)/dt = ...
            
            # No implementation so far
            data_grad_f = data
        elif self.example_number == 8: # Wave equation in 1D
            # Derivatives: du(x,t)/dx = -sin(x)*cos(t)-2*sin(2x)*sin(2t)
            #              du(x,t)/dt = -cos(x)*sin(t)+2*cos(2x)*cos(2t) + 2
            partial_x = -np.sin(data[0])*np.cos(data[1]) - 2*np.sin(2*data[0])*np.sin(2*data[1])
            partial_y = -np.cos(data[0])*np.sin(data[1]) + 2*np.cos(2*data[0])*np.cos(2*data[1]) + 2
            data_grad_f = np.vstack((partial_x,partial_y))
        elif self.example_number == 9: # Inhom. wave equation in 1D
            # Derivatives: du(x,t)/dx = exp(x-2t)
            #              du(x,t)/dt = -2*exp(x-2t)
            partial_x = np.exp(data[0]-2*data[1])
            partial_y = np.exp(data[0]-2*data[1]) * (-2)
            data_grad_f = np.vstack((partial_x,partial_y))
        return data_grad_f
    
