import torch
import scipy
import numpy as np
import networkx as nx
from utils import dual_grad_linear_regression, create_pi_ij, compute_resistance_edge


class Optimizer(object):
    def __init__(
            self, f, data, labels, asynchronous=False, stochastic=False, batch_size=1
    ):
        """
        Initialize the optimizer.
        
        Parameters:
            - f (nn.Module): the convex function to optimize.
            - data (torch.tensor): of shape [n_workers, n_data_per_worker, dim]
                                   the data points at each worker.
            - labels (torch.tensor): of shape [n_workers, n_data_per_worker, 1]
                                     the labels of each data point.
            - asynchronous (bool): whether or not we are in an asynchronous case.
            - stochastic (bool): whether or not use stochastic gradients.
            - batch_size (int): the mini batch size to use if stochastic.
        """

        self.f = f
        self.data = data
        self.labels = labels
        self.asynchronous = asynchronous
        self.stochastic = stochastic

    def compute_grads(self, i=None):
        """
        Compute the grads $\partial f_i / \partial x_i$
        by performing a forward and backward pass with the loss
        $f = \sum_i f_i$ in the syncrhonous regime, or with the loss
        $f = f_i$ in the asynchronous one.
        """
        self.f.zero_grad()
        if self.asynchronous:
            if type(i) is not int:
                raise ValueError(
                    "In the asynchronous regime, you should provide the index of the local function you want to differentiate."
                )
            data = self.data  # shape [n_workers, n_data_per_worker, dim]
            labels = self.labels  # shape [n_workers, n_data_per_worker, 1]
            # if we use SGD, we draw samples uniformly at random from the local dataset
            # and only compute the loss with respect to these samples
            if self.stochastic:
                id_samples = np.random.randint(0, self.data.shape[1], batch_size)
                data = data[:, id_samples, :]  # shape [n_workers, 1, dim]
                labels = labels[:, id_samples, :]  # shape [n_workers, 1, 1]
            # In the asynchronous case, we only compute the gradient of f_i
            loss = self.f(data, labels)[i]
        else:
            # In the synchronous case, we compute the gradients of every f_i
            loss = torch.sum(self.f(self.data, self.labels))
        loss.backward()

    def get_grads(self, ):
        """
        Returns the gradients and current parameters of f_i
        
        Returns:
            - X (torch.tensor): of shape [n_workers, dim, 1],
                                the current parameters of each f_i.
            - grad_X (torch.tensor): of shape [n_workers, dim, 1]
                                     the gradients \partial f_i / \partial x_i.
        """

        # loop of length 1: there is only 1 set of parameters in matrix form,
        # see the definition of the functions in cvx_functions.py
        for xi in self.f.parameters():
            X = xi.data.squeeze()
            # in the asynchronous case, grads are 0 everywhere exept at the index i
            grad_X = xi.grad.data.squeeze()
        return X, grad_X

    def update_params_f(self, X_update):
        """
        Replace the previous parameters with X_update.
        """
        for xi in self.f.parameters():
            xi.data = X_update.detach().clone().unsqueeze(-1)

    def initialize(self, ):
        raise NotImplementedError

    def step(self, ):
        raise NotImplementedError


class DADAO_optimizer(Optimizer):
    """
    Implementation of our decentralized asynchronous method with decoupled gradient and consensus steps.
    See ( https://hal.archives-ouvertes.fr/hal-03737694/document ) for more details.
    """

    def __init__(
            self,
            f,
            data,
            labels,
            chi_1_star,
            lamb_mix,
            n_nodes,
            mu=1,
            L=1,
            stochastic=False,
            batch_size=1,
    ):
        super().__init__(
            f,
            data,
            labels,
            asynchronous=True,
            stochastic=stochastic,
            batch_size=batch_size,
        )
        """
        Parameters:
            - f (nn.Module): the convex function to optimize.
            - data (torch.tensor): of shape [n_workers, n_data_per_worker, dim]
                                   the data points at each worker.
            - labels (torch.tensor): of shape [n_workers, n_data_per_worker, 1]
                                     the labels of each data point.
            - n_nodes (int): the number of workers.
            - rho (float): the graph condition number.
            - mu (float): the strong convexity coefficient.
            - L (float): the smoothness coefficient.
        """

        self.L = L
        self.mu = mu
        self.chi_1_star = chi_1_star
        self.lamb_mix = lamb_mix
        self.n_nodes = n_nodes

        self.initialize()

    def initialize(self, ):

        # the parameters X of the convex functions f_i
        # are already initialize randomly by Pytorch.
        # compute the gradients at this initialization point.
        self.compute_grads(
            i=0
        )  # put an arbitrary index, anyway we do not gather the gradients here
        X, _ = self.get_grads()
        self.X = X.detach().clone()
        # Initialize the variables  at zero
        self.Y = torch.zeros(X.shape).double()
        self.Z = torch.zeros(X.shape).double()
        self.X_tilde = torch.zeros(X.shape).double()
        self.Y_tilde = torch.zeros(X.shape).double()
        self.Z_tilde = torch.zeros(X.shape).double()
        # COMPUTE ALL THE CONSTANTS
        mu = self.mu
        nu = mu / 2
        L = self.L
        self.nu = nu
        # Constants for the ODE
        alpha = (1 / 4) * np.sqrt(nu / L)
        alpha_tilde = (1 / 8) * np.sqrt(nu / L)
        eta = (1 / 8) * np.sqrt(nu / L)
        eta_tilde = eta
        theta = (1 / 2) * np.sqrt(L / nu)
        # Constants for the jumps
        self.beta = 1 / 2
        self.beta_tilde = (2 * (self.chi_1_star)) * np.sqrt(L / nu)
        self.delta = (1 / 4) * np.sqrt(nu / L)
        self.delta_tilde = 1
        self.gamma = 1 / (4 * L)
        self.gamma_tilde = 1 / (4 * np.sqrt(L * nu))

        # Matrix describing the linear dynamic
        ode_matrix = torch.zeros((6, 6))
        ode_matrix[0, 0] = -eta
        ode_matrix[0, 1] = eta
        ode_matrix[1, 0] = eta_tilde
        ode_matrix[1, 1] = -eta_tilde
        ode_matrix[2, 2] = -alpha
        ode_matrix[2, 3] = alpha
        ode_matrix[3, 1] = -theta * nu
        ode_matrix[3, 2] = -theta
        ode_matrix[3, 4] = -theta
        ode_matrix[4, 4] = -alpha
        ode_matrix[4, 5] = alpha
        ode_matrix[5, 4] = alpha_tilde
        ode_matrix[5, 5] = -alpha_tilde

        self.ode_matrix = ode_matrix.double()

        # Initialize the list of times of last event at each node
        self.t_last_event = np.zeros(self.n_nodes)

    def update_params_i_after_ode(self, t_old, t_new, i):
        with torch.no_grad():
            # first, gather all the params at node i in one tensor
            X_i = self.X[i].unsqueeze(0)
            X_tilde_i = self.X_tilde[i].unsqueeze(0)
            Y_i = self.Y[i].unsqueeze(0)
            Y_tilde_i = self.Y_tilde[i].unsqueeze(0)
            Z_i = self.Z[i].unsqueeze(0)
            Z_tilde_i = self.Z_tilde[i].unsqueeze(0)
            Params_i = torch.cat(
                [X_i, X_tilde_i, Y_i, Y_tilde_i, Z_i, Z_tilde_i], dim=0
            )
            # Compute the exponential of the matrix of the ode system
            exp_M = torch.linalg.matrix_exp(self.ode_matrix * (t_new - t_old))
            # Do the mixing
            Params_i_new = exp_M @ Params_i
            # Update the params in memomy
            self.X[i] = Params_i_new[0]
            self.X_tilde[i] = Params_i_new[1]
            self.Y[i] = Params_i_new[2]
            self.Y_tilde[i] = Params_i_new[3]
            self.Z[i] = Params_i_new[4]
            self.Z_tilde[i] = Params_i_new[5]
            # Update params in the function
            self.update_params_f(self.X)

    def step(self, event_is_grad, t_event, edge_ij=None, node_i=None):
        """
        Perform one step of the optimizer, whether it is a gradient or communication step.
        
        Parameters:
            - event_is_grad (bool): whether or not the event is a gradient step.
            - t_event (float): the time of the event.
            - edge_ij (tuple): the edge along which communication is happening.
            - node_i (int): the id of the node taking a gradient step.
        """
        # If we take a gradient step
        if event_is_grad:
            if node_i is None:
                raise ValueError(
                    "The index i of the node must be given to take a local gradient step."
                )
            i = node_i
            t_old = self.t_last_event[i]
            # Update params from the ODE
            self.update_params_i_after_ode(t_old, t_event, i)
            # Compute and get the gradients at X_{T_{k+1}-} at node i
            self.compute_grads(i)
            # As we only computed the grads at coordinate i,
            # grad_X is 0 everywhere except at i
            _, grad_X = self.get_grads()
            X_after_ode = self.X.detach().clone()
            X_tilde_after_ode = self.X_tilde.detach().clone()
            Y_tilde_after_ode = self.Y_tilde.detach().clone()
            # perform the gradient steps on X, X_tilde and Y_tilde
            self.X[i] = X_after_ode[i] - self.gamma * (
                    grad_X[i] - self.nu * X_after_ode[i] - Y_tilde_after_ode[i]
            )
            self.X_tilde[i] = X_tilde_after_ode[i] - self.gamma_tilde * (
                    grad_X[i] - self.nu * X_after_ode[i] - Y_tilde_after_ode[i]
            )
            self.Y_tilde[i] = Y_tilde_after_ode[i] + (self.delta + self.delta_tilde) * (
                    grad_X[i] - self.nu * X_after_ode[i] - Y_tilde_after_ode[i]
            )
            # Update params in the function
            self.update_params_f(self.X)
            # Update the list of time of last event
            self.t_last_event[i] = t_event
        # If we mix the parameters along an edge of the communication network
        else:
            if edge_ij is None:
                raise ValueError(
                    "The edge (ij) must be given to perform a communication step."
                )
            i, j = edge_ij[0], edge_ij[1]
            # Get last event times on both nodes
            t_old_i = self.t_last_event[i]
            t_old_j = self.t_last_event[j]
            # Update params from the ODE on both nodes
            self.update_params_i_after_ode(t_old_i, t_event, i)
            self.update_params_i_after_ode(t_old_j, t_event, j)
            # perform the communication steps using gossip matrix pi_ij
            pi_ij = create_pi_ij(i, j, self.n_nodes)
            Z_after_ode = self.Z.detach().clone()
            Z_tilde_after_ode = self.Z_tilde.detach().clone()
            # compute the two messagges transmited
            proj_z = pi_ij @ (self.Y + Z_after_ode)
            # update the params on both nodes
            for k in [i, j]:
                self.Z[k] = Z_after_ode[k] - self.beta * proj_z[k]
                self.Z_tilde[k] = Z_tilde_after_ode[k] - self.beta_tilde * proj_z[k]
            # Update the list of time of last event on both nodes
            self.t_last_event[i] = t_event
            self.t_last_event[j] = t_event


class ADOMplus_optimizer(Optimizer):
    """
    Implementation of the ADOM+ optimizer ( https://openreview.net/forum?id=L8-54wkift )
    """

    def __init__(self, f, data, labels, mu=1, L=1, chi=1):
        super().__init__(f, data, labels)
        """"
        Parameters:
            - f (nn.Module): the convex function to optimize.
            - data (torch.tensor): of shape [n_workers, n_data_per_worker, dim]
                                   the data points at each worker.
            - labels (torch.tensor): of shape [n_workers, n_data_per_worker, 1]
                                     the labels of each data point.
            - mu (float): the strong convexity coefficient.
            - L (float): the smoothness coefficient.
            - chi (float): for the matrices considers, equals \frac{\lambda_max}{\lambda_min^+}
                           of the graph's Laplacians, see page 4 of the paper.
        """

        self.L = L
        self.mu = mu
        self.chi = chi

        self.initialize()

    def initialize(self, ):
        self.compute_grads()
        X, grad_X = self.get_grads()
        self.X = X
        self.grad_X_g = grad_X

        self.X = torch.ones(X.shape, dtype=torch.float64)
        self.M = torch.randn(X.shape, dtype=torch.float64)
        self.Y = torch.randn(X.shape, dtype=torch.float64)
        self.Z = torch.zeros(X.shape, dtype=torch.float64)

        self.X_f = self.X
        self.Y_f = self.Y
        self.Z_f = self.Z
        # Compute the constants
        self.tau_1 = 1 / (np.sqrt(self.L / self.mu) + 0.5)
        self.tau_2 = np.sqrt(self.mu / self.L)
        self.sigma_2 = self.tau_2 / (16 * self.chi)
        self.sigma_1 = 1 / ((1 / self.sigma_2) + 0.5)
        self.eta = 1 / (self.L * self.tau_2)
        self.alpha = self.mu / 2
        self.beta = 1 / (2 * self.L)
        self.delta = 1 / (17 * self.L)
        self.nu = self.mu / 2
        self.gamma = self.nu / (14 * self.sigma_2 * (self.chi ** 2))
        self.theta = self.nu / (4 * self.sigma_2)
        self.zeta = 1 / 2

    def step(self, W):
        # compute X_g
        X_update = self.tau_1 * self.X + (1 - self.tau_1) * self.X_f
        self.update_params_f(X_update)
        # compute and get the gradients at X_g
        self.compute_grads()
        X_g, self.grad_X_g = self.get_grads()
        # Compute Y_g and Z_g
        Y_g = self.sigma_1 * self.Y + (1 - self.sigma_1) * self.Y_f
        Z_g = self.sigma_1 * self.Z + (1 - self.sigma_1) * self.Z_f
        # compute the constants in the updates of Y and X
        c_1 = (1 + self.eta * self.alpha) / (
                (1 + self.theta * self.beta) * (1 + self.eta * self.alpha)
                + self.theta * self.eta
        )
        c_2 = self.theta * self.beta + self.eta * (
                self.theta / (1 + self.eta * self.alpha)
        )
        c_3 = self.theta * (
                self.beta * self.nu
                + self.eta * (self.nu + self.alpha) / (1 + self.eta * self.alpha)
        )
        c_4 = self.theta / (1 + self.eta * self.alpha)
        c_5 = 1 / (1 + self.eta * self.alpha)
        # Update X, Y, Z and M
        Y_new = c_1 * (
                self.Y
                + self.grad_X_g * c_2
                - X_g * c_3
                - self.X * c_4
                - (self.theta / self.nu) * (Y_g + Z_g)
        )
        X_new = c_5 * (
                self.X
                + self.eta * self.alpha * X_g
                - self.eta * (self.grad_X_g - self.nu * X_g - Y_new)
        )
        Z_new = (
                self.Z
                + self.gamma * self.delta * (Z_g - self.Z)
                - (torch.eye(W.shape[0]) - W)
                @ ((self.gamma / self.nu) * (Y_g + Z_g) + self.M)
        )
        # W is equal to I_n - L, with L a gossip matrix
        M_new = W @ ((self.gamma / self.nu) * (Y_g + Z_g) + self.M)
        # Compute X_f, Y_f, Z_f
        self.X_f = X_g + self.tau_2 * (X_new - self.X)
        self.Y_f = Y_g + self.sigma_2 * (Y_new - self.Y)
        self.Z_f = Z_g - self.zeta * (torch.eye(W.shape[0]) - W) @ (Y_g + Z_g)
        # replace the parameters by their new version
        self.Y = Y_new
        self.X = X_new
        self.Z = Z_new
        self.M = M_new
        self.update_params_f(self.X)


class ADOMplusVR_optimizer(Optimizer):
    """
    Implementation of the ADOM+VR optimizer
    """

    def __init__(self, f, data, labels, dataw, mu=1, L=1, chi=1, batch_size=10, ):
        super().__init__(f, data, labels)
        """"
        Parameters:
            - f (nn.Module): the convex function to optimize.
            - data (torch.tensor): of shape [n_workers, n_data_per_worker, dim]
                                   the data points at each worker.
            - labels (torch.tensor): of shape [n_workers, n_data_per_worker, 1]
                                     the labels of each data point.
            - mu (float): the strong convexity coefficient.
            - L (float): the smoothness coefficient.
            - chi (float): for the matrices considers, equals \frac{\lambda_max}{\lambda_min^+}
                           of the graph's Laplacians, see page 4 of the paper.
        """
        self.dataw = dataw
        self.batch_size = batch_size
        self.L = L
        self.mu = mu
        self.chi = chi
        self.asynchronous = False
        self.initialize()

    def compute_full_grads(self, ):
        """
        Compute the grads $\partial f_i / \partial x_i$
        by performing a forward and backward pass with the loss
        $f = \sum_i f_i$ in the syncrhonous regime, or with the loss
        $f = f_i$ in the asynchronous one.
        """
        self.f.zero_grad()

        data = self.data  # shape [n_workers, n_data_per_worker, dim]
        labels = self.labels  # shape [n_workers, n_data_per_worker, 1]

        loss = torch.sum(self.f(data, labels))
        loss.backward()

    def compute_grads(self, ):
        """
        Compute the grads $\partial f_i / \partial x_i$
        by performing a forward and backward pass with the loss
        $f = \sum_i f_i$ in the syncrhonous regime, or with the loss
        $f = f_i$ in the asynchronous one.
        """
        self.f.zero_grad()

        data = self.data  # shape [n_workers, n_data_per_worker, dim]
        dataw = self.dataw  # shape [n_workers, n_data_per_worker, dim]
        labels = self.labels  # shape [n_workers, n_data_per_worker, 1]

        idx = np.random.randint(0, self.data.shape[1], self.batch_size)

        data = data[:, idx, :]  # shape [n_workers, batch_size_data_per_worker, dim]
        dataw = dataw[:, idx, :]  # shape [n_workers, batch_size_data_per_worker, dim]
        labels = labels[:, idx, :]  # shape [n_workers, batch_size_data_per_worker, 1]

        loss = (torch.sum(self.f(data, labels)) - torch.sum(self.f(dataw, labels))) / self.batch_size
        loss.backward()

    def compute_grads_w(self, ):
        self.f.zero_grad()
        loss = torch.sum(self.f(self.dataw, self.labels))
        loss.backward()

    def get_grad_w(self, ):
        for xi in self.f.parameters():
            w = xi.data.squeeze()
            grad_w = xi.grad.data.squeeze()
        return w, grad_w

    def update_params_f_x(self, X):
        """
        Replace the previous parameters with X_update.
        """
        for xi in self.f.parameters():
            xi.data = X.detach().clone().unsqueeze(-1)

    def update_params_f_w(self, w):
        for xi in self.f.parameters():
            xi.dataw = w.detach().clone().unsqueeze(-1)

    def initialize(self, ):
        n = self.data.shape[1]
        self.b = np.max([np.sqrt(n), n * np.sqrt(self.mu / self.L)])
        self.batch_size = np.int64(self.b)

        # print("self.b", self.b)

        # self.compute_full_grads()
        # X, grad_X = self.get_grads()
        # self.X = X
        # self.grad_batch = grad_X

        self.X = torch.ones(self.data.shape[0], self.data.shape[2], dtype=torch.float64)
        # self.X = torch.ones(X.shape,  dtype=torch.float64)
        self.M = torch.randn(self.X.shape, dtype=torch.float64)
        self.Y = torch.randn(self.X.shape, dtype=torch.float64)
        self.Z = torch.zeros(self.X.shape, dtype=torch.float64)
        self.w = torch.zeros(self.X.shape, dtype=torch.float64)

        self.X_f = self.X
        self.Y_f = self.Y
        self.Z_f = self.Z

        # Compute the constants
        b = torch.tensor(self.b)

        self.tau_0 = 1 / (2 * b)
        self.tau_2 = np.min([1 / 2, np.max([1, np.sqrt(n) / b]) * np.sqrt(self.mu / self.L)])
        self.tau_1 = (1 - self.tau_0) / (1 / self.tau_2 + 1 / 2)

        self.alpha = 1/10 #self.mu
        self.nu = self.mu / 4

        self.sigma_2 = np.sqrt(self.mu) / (16 * self.chi * np.sqrt(self.L))
        self.sigma_1 = 1 / (1 / self.sigma_2 + 1 / 2)

        self.eta = 1 / (self.L * (self.tau_2 + 2 * self.tau_1 / (1 - self.tau_1)))

        self.beta = 1 / (2 * self.L)
        self.delta = 1 / (17 * self.L)

        self.gamma = self.nu / (14 * self.sigma_2 * np.square(self.chi))

        self.theta = self.nu / (4 * self.sigma_2)

        self.zeta = 1 / 4  # self.mu

        self.Lambda = n / b * (1 / 2 + 1 / (b * self.tau_1))

        self.p_1 = 1 / (1 * self.Lambda)
        self.p_2 = 1 / (self.Lambda * b * self.tau_1)

        # print("self.p_1", self.p_1)
        # print("self.p_2", self.p_2)

        self.update_params_f_w(self.w)
        self.compute_grads_w()
        self.w, self.grad_w = self.get_grad_w()
        self.r = -1

    def step(self, W):
        # compute X_g
        X_update = self.tau_1 * self.X + self.tau_0 * self.w + (1 - self.tau_1 - self.tau_0) * self.X_f
        self.update_params_f_x(X_update)

        # compute and get the gradients at X_g
        self.compute_grads()
        X_g, self.grad_batch = self.get_grads()
        self.grad_batch += self.grad_w

        # Compute Y_g and Z_g
        Y_g = self.sigma_1 * self.Y + (1 - self.sigma_1) * self.Y_f
        Z_g = self.sigma_1 * self.Z + (1 - self.sigma_1) * self.Z_f

        ar = [self.X_f, X_g, self.w]
        self.r = np.random.choice([0, 1, 2], 1, p=[self.p_1, self.p_2, 1 - self.p_1 - self.p_2])[0]
        self.w = ar[self.r]

        if self.r != 2:
            self.update_params_f_w(self.w)
            self.compute_grads_w()
            self.w, self.grad_w = self.get_grad_w()

        # compute the constants in the updates of Y and X
        c_1 = (1 + self.eta * self.alpha) / (
                (1 + self.theta * self.beta) * (1 + self.eta * self.alpha)
                + self.theta * self.eta
        )

        Y_new = c_1 * (
                self.Y
                + self.grad_batch * (
                        self.theta * self.beta + self.eta * (self.theta / (1 + self.eta * self.alpha))
                )
                - X_g * self.theta * (
                        self.beta * self.nu
                        + self.eta * (self.nu + self.alpha) / (1 + self.eta * self.alpha)
                )
                - self.X * self.theta / (1 + self.eta * self.alpha)
                - (self.theta / self.nu) * (Y_g + Z_g)
        )

        X_new = (
                        self.X
                        + self.eta * self.alpha * X_g
                        - self.eta * (self.grad_batch - self.nu * X_g - Y_new)
                ) / (1 + self.eta * self.alpha)

        Z_new = (
                self.Z
                + self.gamma * self.delta * (Z_g - self.Z)
                - (torch.eye(W.shape[0]) - W)
                @ ((self.gamma / self.nu) * (Y_g + Z_g) + self.M)  # ok
        )

        M_new = (
                self.gamma / self.nu * (Y_g + Z_g) + self.M
                - (torch.eye(W.shape[0]) - W) @ ((self.gamma / self.nu) * (Y_g + Z_g) + self.M)
        )
        # Compute X_f, Y_f, Z_f
        self.X_f = X_g + self.tau_2 * (X_new - self.X)
        self.Y_f = Y_g + self.sigma_2 * (Y_new - self.Y)
        self.Z_f = Z_g - self.zeta * (torch.eye(W.shape[0]) - W) @ (Y_g + Z_g)
        # replace the parameters by their new version
        self.Y = Y_new
        self.X = X_new
        self.Z = Z_new
        self.M = M_new
        self.update_params_f(self.X)


class GTPAGE_optimizer(Optimizer):
    """
    Implementation of the BEER+VR optimizer
    """

    def __init__(self, f, data, labels, mu=1, L=1, chi=1, batch_size=10, ):
        super().__init__(f, data, labels)
        """"
        Parameters:
            - f (nn.Module): the convex function to optimize.
            - data (torch.tensor): of shape [n_workers, n_data_per_worker, dim]
                                   the data points at each worker.
            - labels (torch.tensor): of shape [n_workers, n_data_per_worker, 1]
                                     the labels of each data point.
            - mu (float): the strong convexity coefficient.
            - L (float): the smoothness coefficient.
            - chi (float): for the matrices considers, equals \frac{\lambda_max}{\lambda_min^+}
                           of the graph's Laplacians, see page 4 of the paper.
        """
        self.L = L
        self.m = self.data.shape[0]
        # print("self.data.shape[0]", self.data.shape)
        self.n = self.data.shape[1]
        # print("self.batch_size", self.batch_size)
        self.mu = mu
        self.chi = chi
        self.asynchronous = False
        self.f = f
        self.r = -1

        self.initialize()

    def initialize(self, ):
        self.X = torch.randn(self.data.shape[0], self.data.shape[2], dtype=torch.float64)
        self.update_params_f(self.X)
        self.compute_full_grads()
        _, grad_X = self.get_grads()
        # self.X = _#torch.ones(_.shape,  dtype=torch.float64)
        self.Y = grad_X

        self.eta = 1e-2
        self.V = self.Y / self.m
        self.batch_size = 50#np.int64(np.sqrt(self.n) * self.L)
        self.p = self.batch_size / (self.batch_size + self.n)
        # print("p",self.p)
        # self.grad_previous = grad_X

    def update_params_f(self, X):
        """
        Replace the previous parameters with X_update.
        """
        for xi in self.f.parameters():
            xi.data = X.detach().clone().unsqueeze(-1)

    def get_grads(self, ):
        """
        Returns the gradients and current parameters of f_i

        Returns:
            - X (torch.tensor): of shape [n_workers, dim, 1],
                                the current parameters of each f_i.
            - grad_X (torch.tensor): of shape [n_workers, dim, 1]
                                     the gradients \partial f_i / \partial x_i.
        """

        # loop of length 1: there is only 1 set of parameters in matrix form,
        # see the definition of the functions in cvx_functions.py
        for xi in self.f.parameters():
            X = xi.data.squeeze()
            # in the asynchronous case, grads are 0 everywhere exept at the index i
            grad_X = xi.grad.data.squeeze()
        return X, grad_X

    def compute_full_grads(self, ):
        self.f.zero_grad()
        data = self.data  # shape [n_workers, n_data_per_worker, dim]
        labels = self.labels  # shape [n_workers, n_data_per_worker, 1]
        loss = torch.sum(self.f(data, labels))
        loss.backward()

    def compute_grads_f(self, idx, X):
        for xi in self.f.parameters():
            xi.data = X.detach().clone().unsqueeze(-1)

        self.f.zero_grad()

        data = self.data[:, idx, :]  # shape [n_workers, batch_size_data_per_worker, dim]
        labels = self.labels[:, idx, :]  # shape [n_workers, batch_size_data_per_worker, 1]

        loss = torch.sum(self.f(data, labels)) / self.batch_size
        loss.backward()

    def compute_grads_f2(self, X):
        for xi in self.f.parameters():
            xi.data = X.detach().clone().unsqueeze(-1)
        self.f.zero_grad()
        loss = torch.sum(self.f(self.data, self.labels))
        loss.backward()

    def getIdx(self, ):
        return np.random.randint(0, self.data.shape[1], self.batch_size)

    def step(self, W):
        X_new = (torch.eye(W.shape[0]) - W) @ self.X - self.eta * self.V

        self.r = np.random.choice([0, 1], 1, p=[self.p, 1 - self.p])[0]

        if self.r == 0:
            self.update_params_f(X_new)
            self.compute_full_grads()
            _, grad = self.get_grads()
            Y_new = grad
        else:
            idx = self.getIdx()
            self.compute_grads_f(idx, X_new)
            _, grads_x_new = self.get_grads()

            self.compute_grads_f(idx, self.X)
            _, grads_x = self.get_grads()

            Y_new = self.Y + grads_x_new - grads_x

        V_new = (torch.eye(W.shape[0]) - W) @ self.V + Y_new - self.Y

        self.Y = Y_new.detach().clone()
        self.V = V_new.detach().clone()
        self.X = X_new.detach().clone()


class GT_SARAH_optimizer(Optimizer):
    """
    Implementation of the BEER+VR optimizer
    """

    def __init__(self, f, data, labels, dataw, mu=1, L=1, chi=1, batch_size=10, ):
        super().__init__(f, data, labels)
        """"
        Parameters:
            - f (nn.Module): the convex function to optimize.
            - data (torch.tensor): of shape [n_workers, n_data_per_worker, dim]
                                   the data points at each worker.
            - labels (torch.tensor): of shape [n_workers, n_data_per_worker, 1]
                                     the labels of each data point.
            - mu (float): the strong convexity coefficient.
            - L (float): the smoothness coefficient.
            - chi (float): for the matrices considers, equals \frac{\lambda_max}{\lambda_min^+}
                           of the graph's Laplacians, see page 4 of the paper.
        """
        self.dataw = dataw
        self.batch_size = batch_size
        self.L = L
        self.mu = mu
        self.chi = chi
        self.asynchronous = False
        self.f = f

        self.eta = 0.1
        self.n_inner_iters = 10

        # print(self.data.shape)
        # print("self.X.shape", self.X.shape)
        self.v = torch.zeros((self.data.shape[0], self.data.shape[2]), dtype=torch.float64)
        self.y = torch.zeros((self.data.shape[0], self.data.shape[2]), dtype=torch.float64)
        self.X = torch.randn((self.data.shape[0], self.data.shape[2]), dtype=torch.float64)

    def compute_grads(self, ):

        self.f.zero_grad()

        data = self.data  # shape [n_workers, n_data_per_worker, dim]
        labels = self.labels  # shape [n_workers, n_data_per_worker, 1]

        idx = np.random.randint(0, self.data.shape[1], self.batch_size)

        data = data[:, idx, :]  # shape [n_workers, batch_size_data_per_worker, dim]
        labels = labels[:, idx, :]  # shape [n_workers, batch_size_data_per_worker, 1]

        loss = torch.sum(self.f(data, labels)) / self.batch_size
        loss.backward()

    def compute_full_grads(self, ):
        self.f.zero_grad()
        data = self.data  # shape [n_workers, n_data_per_worker, dim]
        labels = self.labels  # shape [n_workers, n_data_per_worker, 1]
        loss = torch.sum(self.f(data, labels))
        loss.backward()

    def update_params_f(self, X):
        """
        Replace the previous parameters with X_update.
        """
        for xi in self.f.parameters():
            xi.data = X.detach().clone().unsqueeze(-1)

    def get_grads(self, ):
        """
        Returns the gradients and current parameters of f_i

        Returns:
            - X (torch.tensor): of shape [n_workers, dim, 1],
                                the current parameters of each f_i.
            - grad_X (torch.tensor): of shape [n_workers, dim, 1]
                                     the gradients \partial f_i / \partial x_i.
        """

        # loop of length 1: there is only 1 set of parameters in matrix form,
        # see the definition of the functions in cvx_functions.py
        for xi in self.f.parameters():
            X = xi.data.squeeze()
            # in the asynchronous case, grads are 0 everywhere exept at the index i
            grad_X = xi.grad.data.squeeze()
        return X, grad_X

    def step(self, W):

        self.v_last = self.v
        self.x_last = self.X

        self.compute_full_grads()
        _, grad_X = self.get_grads()
        self.grad_X = grad_X

        self.v = self.grad_X
        self.y = W @ self.y + self.v - self.v_last
        self.X = W @ self.X - self.eta * self.y

        for inner_iter in range(self.n_inner_iters):
            self.update_params_f(self.X)
            self.compute_grads()
            _, grad_x = self.get_grads()

            self.update_params_f(self.x_last)
            self.compute_grads()
            _, grad_x_last = self.get_grads()

            self.v_last = self.v
            self.x_last = self.X

            self.v = self.v + grad_x - grad_x_last

            self.y = W @ self.y + self.v - self.v_last
            self.X = W @ self.X - self.eta * self.y


class AccGT_optimizer(Optimizer):
    """
    Implementation of the AccGT optimizer
    """

    def __init__(self, f, data, labels, dataw, mu=1, L=1, chi=1, batch_size=10, ):
        super().__init__(f, data, labels)
        """"
        Parameters:
            - f (nn.Module): the convex function to optimize.
            - data (torch.tensor): of shape [n_workers, n_data_per_worker, dim]
                                   the data points at each worker.
            - labels (torch.tensor): of shape [n_workers, n_data_per_worker, 1]
                                     the labels of each data point.
            - mu (float): the strong convexity coefficient.
            - L (float): the smoothness coefficient.
            - chi (float): for the matrices considers, equals \frac{\lambda_max}{\lambda_min^+}
                           of the graph's Laplacians, see page 4 of the paper.
        """
        self.dataw = dataw
        self.batch_size = batch_size
        self.L = L
        self.mu = mu
        self.chi = chi
        self.asynchronous = False
        self.f = f

        self.eta = 0.1
        self.batch_size = batch_size
        self.initialize()

    def compute_grads(self, ):
        self.f.zero_grad()
        loss = torch.sum(self.f(self.data, self.labels))
        loss.backward()

    def initialize(self, ):
        self.n = self.data.shape[0]
        self.I_n = torch.eye(self.n)

        self.compute_grads()
        X, grad = self.get_grads()

        # self.X = X
        self.X = torch.ones(X.shape, dtype=torch.float64)
        self.Y = X
        self.Z = X
        self.s_previous = grad
        self.grad_previous = grad

        self.alpha = 1e-3
        self.theta = np.sqrt(self.mu * self.alpha) / 2
        # self.theta = 1.

    def step(self, W):
        self.Y = self.theta * self.Z + (1 - self.theta) * self.X

        self.update_params_f(self.Y)
        self.compute_grads()
        X, grad = self.get_grads()

        s = W @ self.s_previous + grad - self.grad_previous

        self.Z = 1 / (1 + self.mu * self.alpha / self.theta) * (
                W @ (self.mu * self.alpha * self.Y / self.theta + self.Z) - self.alpha * s / self.theta
        )
        self.X = self.theta * self.Z + (1 - self.theta) * W @ self.X
        # self.theta = 0.5 * (np.sqrt(self.theta**4 + 4 * self.theta**2) - self.theta**2)

        self.grad_previous = grad
        self.s_previous = s


class AccVRExtra_optimizer(Optimizer):
    """
    Implementation of the Acc_VR_Extra optimizer
    """

    def __init__(self, f, data, labels, dataw, mu=1, L=1, chi=1, batch_size=10, ):
        super().__init__(f, data, labels)
        """"
        Parameters:
            - f (nn.Module): the convex function to optimize.
            - data (torch.tensor): of shape [n_workers, n_data_per_worker, dim]
                                   the data points at each worker.
            - labels (torch.tensor): of shape [n_workers, n_data_per_worker, 1]
                                     the labels of each data point.
            - mu (float): the strong convexity coefficient.
            - L (float): the smoothness coefficient.
            - chi (float): for the matrices considers, equals \frac{\lambda_max}{\lambda_min^+}
                           of the graph's Laplacians, see page 4 of the paper.
        """
        self.dataw = dataw
        self.batch_size = batch_size
        self.L = L
        self.mu = mu
        self.chi = chi
        self.asynchronous = False
        self.f = f

        self.initialize()

    def initialize(self, ):
        self.n = self.data.shape[0]
        self.X = torch.ones(self.data.shape[0], self.data.shape[2], dtype=torch.float64)

        self.Z = self.X
        self.w = self.X
        self.lam = torch.ones(self.X.shape, dtype=torch.float64)

        self.alpha = 5e-2
        self.batch_size = np.int64(np.sqrt(self.n))

        self.zeta_1 = 3e-1
        self.zeta_2 = self.L / (2 * self.batch_size)

        self.p = self.batch_size / self.n

        self.update_params_f_w(self.w)
        self.compute_grads_w()
        self.W, self.grad_w = self.get_grad_w()

    def compute_grads(self, ):
        """
        Compute the grads $\partial f_i / \partial x_i$
        by performing a forward and backward pass with the loss
        $f = \sum_i f_i$ in the syncrhonous regime, or with the loss
        $f = f_i$ in the asynchronous one.
        """
        self.f.zero_grad()

        data = self.data  # shape [n_workers, n_data_per_worker, dim]
        dataw = self.dataw  # shape [n_workers, n_data_per_worker, dim]
        labels = self.labels  # shape [n_workers, n_data_per_worker, 1]

        idx = np.random.randint(0, self.data.shape[1], self.batch_size)

        data = data[:, idx, :]  # shape [n_workers, batch_size_data_per_worker, dim]
        dataw = dataw[:, idx, :]  # shape [n_workers, batch_size_data_per_worker, dim]
        labels = labels[:, idx, :]  # shape [n_workers, batch_size_data_per_worker, 1]

        loss = (torch.sum(self.f(data, labels)) - torch.sum(self.f(dataw, labels))) / self.batch_size
        loss.backward()

    def compute_grads_w(self, ):
        self.f.zero_grad()
        loss = torch.sum(self.f(self.dataw, self.labels))
        loss.backward()

    def get_grad_w(self, ):
        for xi in self.f.parameters():
            w = xi.data.squeeze()
            grad_w = xi.grad.data.squeeze()
        return w, grad_w

    def update_params_f(self, X):
        for xi in self.f.parameters():
            xi.data = X.detach().clone().unsqueeze(-1)

    def update_params_f_w(self, w):
        for xi in self.f.parameters():
            xi.dataw = w.detach().clone().unsqueeze(-1)

    def step(self, W):

        self.U = (torch.eye(W.shape[0]) - W) / 2 #torch.sqrt(W / 2)
        Y_new = self.zeta_1 * self.Z + self.zeta_2 * self.w + (1 - self.zeta_1 - self.zeta_2) * self.X

        self.update_params_f(Y_new)
        self.compute_grads()

        _, self.grad_batch = self.get_grads()
        self.grad_batch += self.grad_w

        Z_new = 1 / (1 + self.mu * self.alpha / self.zeta_1) * \
                (
                        self.mu * self.alpha * Y_new / self.zeta_1 +
                        self.Z -
                        (1/self.zeta_1) *
                        (
                                self.alpha * self.grad_batch +
                                self.U @ self.lam +
                                self.zeta_1 * ((torch.eye(W.shape[0]) - W@W)/2) @ self.Z
                        )
                )

        self.lam = self.lam + self.zeta_1 * (self.U @ Z_new)

        X_new = Y_new + self.zeta_1 * (Z_new - self.Z)

        ar = [self.X, self.w]
        self.r = np.random.choice([0, 1], 1, p=[self.p, 1 - self.p])[0]
        self.w = ar[self.r]

        if self.r == 0:
            self.update_params_f_w(self.w)
            self.compute_grads_w()
            self.w, self.grad_w = self.get_grad_w()

        self.Z = Z_new
        self.X = X_new


class Destress_optimizer(Optimizer):
    """
    Implementation of the Destress optimizer
    """

    def __init__(self, f, data, labels, dataw, mu=1, L=1, chi=1, batch_size=10, ):
        super().__init__(f, data, labels)
        """"
        Parameters:
            - f (nn.Module): the convex function to optimize.
            - data (torch.tensor): of shape [n_workers, n_data_per_worker, dim]
                                   the data points at each worker.
            - labels (torch.tensor): of shape [n_workers, n_data_per_worker, 1]
                                     the labels of each data point.
            - mu (float): the strong convexity coefficient.
            - L (float): the smoothness coefficient.
            - chi (float): for the matrices considers, equals \frac{\lambda_max}{\lambda_min^+}
                           of the graph's Laplacians, see page 4 of the paper.
        """
        self.dataw = dataw
        self.batch_size = batch_size
        self.L = L
        self.mu = mu
        self.chi = chi
        self.asynchronous = False
        self.f = f

        self.initialize()

    def T(x, k):
        if k == 0:
            if type(x) is np.ndarray:
                return np.eye(x.shape[0])
            else:
                return 1

        if type(x) is np.ndarray:
            prev = np.eye(x.shape[0])
        else:
            prev = 1

        current = x
        for _ in range(k - 1):
            current, prev = 2 * np.dot(x, current) - prev, current

        return current

    def initialize(self, ):
        self.n = self.data.shape[0]
        self.X = torch.randn(self.data.shape[0], self.data.shape[2]).double()
        self.Y = torch.zeros(self.X.shape).double()

        K_in=1
        K_out=1
        opt=0
        n_inner_iters=100
        eta=0.1
        batch_size=1

        self.K_in = K_in
        self.K_out = K_out

        self.eta = eta
        self.opt = opt
        self.n_inner_iters = n_inner_iters
        self.batch_size = batch_size

        average_matrix = np.ones((self.p.n_agent, self.p.n_agent)) / self.p.n_agent
        alpha = np.linalg.norm(self.W - average_matrix, 2)
        self.W_in = self.T(self.W / alpha, self.K_in) / self.T(1 / alpha, self.K_in)
        self.W_out = self.T(self.W / alpha, self.K_out) / self.T(1 / alpha, self.K_out)

        if len(self.x_0.shape) == 2:
            self.x = np.tile(self.x_0.mean(axis=1), (self.p.n_agent, 1)).T
        else:
            self.x = self.x_0.copy()

        self.grad_last = self.grad(self.x)
        self.s = self.grad_last.copy()
        self.s = np.tile(self.s.mean(axis=1), (self.p.n_agent, 1)).T

    def compute_grads(self, ):
        """
        Compute the grads $\partial f_i / \partial x_i$
        by performing a forward and backward pass with the loss
        $f = \sum_i f_i$ in the syncrhonous regime, or with the loss
        $f = f_i$ in the asynchronous one.
        """
        self.f.zero_grad()

        data = self.data  # shape [n_workers, n_data_per_worker, dim]
        dataw = self.dataw  # shape [n_workers, n_data_per_worker, dim]
        labels = self.labels  # shape [n_workers, n_data_per_worker, 1]

        idx = np.random.randint(0, self.data.shape[1], self.batch_size)

        data = data[:, idx, :]  # shape [n_workers, batch_size_data_per_worker, dim]
        dataw = dataw[:, idx, :]  # shape [n_workers, batch_size_data_per_worker, dim]
        labels = labels[:, idx, :]  # shape [n_workers, batch_size_data_per_worker, 1]

        loss = (torch.sum(self.f(data, labels)) - torch.sum(self.f(dataw, labels))) / self.batch_size
        loss.backward()

    def compute_grads_w(self, ):
        self.f.zero_grad()
        loss = torch.sum(self.f(self.dataw, self.labels))
        loss.backward()

    def get_grad_w(self, ):
        for xi in self.f.parameters():
            w = xi.data.squeeze()
            grad_w = xi.grad.data.squeeze()
        return w, grad_w

    def update_params_f(self, X):
        for xi in self.f.parameters():
            xi.data = X.detach().clone().unsqueeze(-1)

    def update_params_f_w(self, w):
        for xi in self.f.parameters():
            xi.dataw = w.detach().clone().unsqueeze(-1)

    def step(self, W):
        if self.opt == 1:
            n_inner_iters = self.n_inner_iters
        else:
            # Choose random x^{(t)} from n_inner_iters
            n_inner_iters = np.random.randint(1, self.n_inner_iters + 1)
            if type(n_inner_iters) is np.ndarray:
                n_inner_iters = n_inner_iters.item()

        samples = np.random.randint(0, self.p.m, (n_inner_iters, self.p.n_agent, self.batch_size))

        u = self.x.copy()
        v = self.s.copy()
        for inner_iter in range(n_inner_iters):

            u_last, u = u, (u - self.eta * v).dot(self.W_in)
            self.comm_rounds += self.K_in

            v += self.grad(u, j=samples[inner_iter]) - self.grad(u_last, j=samples[inner_iter])
            v = v.dot(self.W_in)
            self.comm_rounds += self.K_in

            # if inner_iter < n_inner_iters - 1:
            #     self.save_metrics(x=u)

        self.x = u

        self.s -= self.grad_last
        self.grad_last = self.grad(self.x)
        self.s += self.grad_last
        self.s = self.s.dot(self.W_out)
        self.comm_rounds += self.K_out


class Continuized_optimizer:
    """
    Implementation of te continuized method with asynchronous *coupled* gradient and communication steps.
    The method is described in page 30-32 of Even et al https://arxiv.org/pdf/2106.07644.pdf
    Note that we corrected a small typo in equation (63) for the method to effectively work
    (the factor 2 should be at the numerator and not the denominator.)
    
    We only implemented the method for the decentralized linear regression case.
    """

    def __init__(self, data, labels, mu, L, mu_gossip, chi_2, G):
        """
        Parameters:
            - data (torch.tensor): of shape [n_workers, n_data_per_worker, dim]
                                   the data points at each worker.
            - labels (torch.tensor): of shape [n_workers, n_data_per_worker, 1]
                                     the labels of each data point.
            - mu (float): the strong convexity coefficient.
            - L (float): the smoothness coefficient.
            - mu_gossip (float): the smallest positive eigenvalue of the graph's
                                 Laplacian considered.
            - chi_2 (float): the graph's worst case resistance.
            - G (nx.graph): the communication graph considered.
        """

        self.L = L
        self.mu = mu
        self.data = data
        self.labels = labels
        self.mu_gossip = mu_gossip
        self.R_over_P = 2 * chi_2
        self.G = G
        self.d = data.shape[-1]
        self.n_workers = data.shape[0]
        self.times = None

        self.initialize()

    def initialize(self, ):
        # Initialize the parameters
        self.Y = torch.zeros((self.n_workers, self.d)).double()
        self.Z = torch.zeros((self.n_workers, self.d)).double()

        # Compute all the constants
        self.eta = np.sqrt(self.mu_gossip / self.R_over_P) * np.sqrt(self.mu / self.L)
        self.eta_tilde = self.eta
        self.gamma = self.mu / (self.R_over_P * 2)
        self.gamma_tilde = np.sqrt(
            (self.L * self.mu) / (2 * self.mu_gossip * self.R_over_P)
        )

        # Matrix describing the linear dynamic
        ode_matrix = torch.zeros((2, 2))
        ode_matrix[0, 0] = -self.eta
        ode_matrix[0, 1] = self.eta
        ode_matrix[1, 0] = self.eta_tilde
        ode_matrix[1, 1] = -self.eta_tilde
        self.ode_matrix = ode_matrix.double()

        # Initialize the list of times of last event at each node
        self.t_last_event = np.zeros(self.n_workers)

        # compute the Laplacian used in the paper.
        # We only consider the case where the probability distribution
        # over the edges is the uniform one.
        L_ = nx.laplacian_matrix(self.G).toarray()
        self.n_edges = len(self.G.edges)
        self.L_norm = L_ / (self.n_edges)

    def update_params_i_after_ode(self, t_old, t_new, i):
        # Update the parameters by integrating the linear ODE.
        with torch.no_grad():
            # first, gather all the params at node i in one tensor
            Y_i = self.Y[i].unsqueeze(0)
            Z_i = self.Z[i].unsqueeze(0)
            Params_i = torch.cat([Y_i, Z_i], dim=0)
            # Compute the exponential of the matrix of the ode system
            exp_M = torch.linalg.matrix_exp(self.ode_matrix * (t_new - t_old))
            # Do the mixing
            Params_i_new = exp_M @ Params_i
            # Update the params in memomy
            self.Y[i] = Params_i_new[0]
            self.Z[i] = Params_i_new[1]

    def step(self, t_event, edge):
        """
        Given the edge for which the P.P.P spiked and the time of the event,
        perform one optimization step, i.e 2 dual gradient steps (one for each node),
        and a communication step.
        """
        i, j = edge[0], edge[1]
        # Get last event times on both nodes
        t_old_i = self.t_last_event[i]
        t_old_j = self.t_last_event[j]
        # Update params from the linear ODE on both nodes
        self.update_params_i_after_ode(t_old_i, t_event, i)
        self.update_params_i_after_ode(t_old_j, t_event, j)
        Y_after_ode = self.Y.detach().clone()
        Z_after_ode = self.Z.detach().clone()
        # compute the two messagges transmited
        grads_i = dual_grad_linear_regression(self.data, self.labels, Y_after_ode[i], i)
        grads_j = dual_grad_linear_regression(self.data, self.labels, Y_after_ode[j], j)
        G_ij = grads_i - grads_j
        # compute the resistance of the edge
        R_ij = compute_resistance_edge(self.L_norm, edge)
        # update the params on both nodes
        self.Y[i] = Y_after_ode[i] - self.gamma * G_ij * (R_ij)
        self.Z[i] = Z_after_ode[i] - (self.gamma_tilde) * G_ij
        self.Y[j] = Y_after_ode[j] + self.gamma * G_ij * (R_ij)
        self.Z[j] = Z_after_ode[j] + (self.gamma_tilde) * G_ij

        # Update the list of time of last event on both nodes
        self.t_last_event[i] = t_event
        self.t_last_event[j] = t_event


class MSDA_optimizer:
    """
    Implementation of the Multi-Step Dual Accelerated (MSDA) method
    from Scaman et al https://arxiv.org/pdf/1702.08704.pdf,
    described in their Algorithm 2.
    
    We implemented the method for the Linear Regression task only.
    """

    def __init__(self, data, labels, mu, L, G):
        """
        Parameters:
            - data (torch.tensor): of shape [n_workers, n_data_per_worker, dim]
                                   the data points at each worker.
            - labels (torch.tensor): of shape [n_workers, n_data_per_worker, 1]
                                     the labels of each data point.
            - mu (float): the strong convexity coefficient.
            - L (float): the smoothness coefficient.
            - G (nx.graph): the communication graph considered.
        """

        self.L = L
        self.alpha = mu
        self.data = data
        self.labels = labels
        self.kappa = L / mu
        self.G = G
        self.n = len(G)
        self.d = data.shape[-1]

        self.initialize()

    def initialize(self, ):

        # Initialize the parameters
        self.X = torch.zeros((self.n, self.d)).double()
        self.Y = torch.zeros((self.n, self.d)).double()
        # compute constants
        # first, compute a graph Laplacian
        L_ = nx.laplacian_matrix(self.G).toarray()
        n_edges = len(self.G.edges)
        # We use the same Laplacian as in the continuized case
        # for fair comparison.
        self.W = torch.tensor(L_ / n_edges).double()
        # compute the eigengap gamma
        lamb_1 = scipy.linalg.eigh(self.W)[0][-1]  # lambda_max
        lamb_min = scipy.linalg.eigh(self.W)[0][1]
        gamma = lamb_min / lamb_1
        alpha = self.alpha
        kappa = self.kappa
        # compute the constants used in the algorihtm
        self.c_1 = (1 - np.sqrt(gamma)) / (1 + np.sqrt(gamma))
        self.c_2 = (1 + gamma) / (1 - gamma)
        self.c_3 = 2 / ((1 + gamma) * lamb_1)
        self.K = int(1 / np.sqrt(gamma))
        self.eta = (alpha * (1 + self.c_1 ** (2 * self.K))) / (
                (1 + self.c_1 ** self.K) ** 2
        )
        self.mu = (
                          (1 + self.c_1 ** self.K) * np.sqrt(kappa) - 1 + self.c_1 ** self.K
                  ) / ((1 + self.c_1 ** self.K) * np.sqrt(kappa) + 1 - self.c_1 ** self.K)

    def accelerated_gossip(self, x, W):
        """
        The accelerated gossip procedure, use a gossip matrix W and
        parameters x.
        """
        # initialize the  variables
        a_k_minus = 1
        a_k = self.c_2
        x_k_minus = x
        matrix = torch.eye(len(self.W)).double() - self.c_3 * self.W
        x_k = self.c_2 * (matrix @ x)
        for k in range(1, self.K):
            a_k_plus = 2 * self.c_2 * a_k - a_k_minus
            x_k_plus = 2 * self.c_2 * (matrix @ x_k) - x_k_minus
            # update the memory
            a_k_minus = a_k.copy()
            x_k_minus = x_k.detach().clone()
            a_k = a_k_plus.copy()
            x_k = x_k_plus.detach().clone()
        return x - x_k / a_k

    def step(self, ):
        # gather the dual gradients from all nodes
        theta = []
        for i in range(self.n):
            theta.append(
                dual_grad_linear_regression(
                    self.data, self.labels, self.X[i], i
                ).unsqueeze(0)
            )
        theta = torch.cat(theta, dim=0)
        # perform one step of the method
        theta_acc = self.accelerated_gossip(theta, self.W)
        Y_plus = self.X.detach().clone() - self.eta * theta_acc
        Y_k = self.Y.detach().clone()
        self.X = (1 + self.mu) * Y_plus - self.mu * Y_k
        self.Y = Y_plus