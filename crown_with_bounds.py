import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from collections import OrderedDict
from contextlib import ExitStack

# JC here you may import any model you wish to verify
# from model import Model
# from simple_model import Model
# from stupid_model import Model
from large_model import Model
import matplotlib.pyplot as plt
from math import ceil


class BoundLinear(nn.Linear):
    def __init(self, in_features, out_features, bias=True):
        super(BoundLinear, self).__init__(in_features, out_features, bias)

    @staticmethod
    def convert(linear_layer):
        r"""Convert a nn.Linear object into a BoundLinear object

        Args: 
            linear_layer (nn.Linear): The linear layer to be converted.
        
        Returns:
            l (BoundLinear): The converted layer
        """ 
        l = BoundLinear(linear_layer.in_features, linear_layer.out_features, linear_layer.bias is not None)
        l.weight.data.copy_(linear_layer.weight.data)
        l.bias.data.copy_(linear_layer.bias.data)
        return l
    
    def bound_backward(self, last_uA, last_lA, start_node=None, optimize=False):
        r"""Backward propagate through the linear layer.

        Args:
            last_uA (tensor): A (the coefficient matrix) that is backward-propagated to this layer
            (from the layers after this layer). It's exclusive for computing the upper bound.

            last_lA (tensor): A that is backward-propagated to this layer. It's exclusive for computing the lower bound.

            start_node (int): An integer indicating the start node of this backward propagation. (It's not used in linear layer)

            optimize (bool): Indicating whether we are optimizing parameters (alpha) (Not used in linear layer)

        Returns:
            uA (tensor): The new A for computing the upper bound after taking this layer into account.
            
            ubias (tensor): The bias (for upper bound) produced by this layer.
            
            lA( tensor): The new A for computing the lower bound after taking this layer into account.
            
            lbias (tensor): The bias (for lower bound) produced by this layer.
        """
        def _bound_oneside(last_A):
            if last_A is None:
                return None, 0
            # propagate A to the nest layer
            next_A = last_A.matmul(self.weight)
            # compute the bias of this layer
            sum_bias = last_A.matmul(self.bias)
            return next_A, sum_bias
        
        # JC note that in the first iteration of backward_range, this is very simple. last_uA=last_lA=I, thus just finding the upper and 
        # lower bound of the output of this linear layer
        uA, ubias = _bound_oneside(last_uA)
        lA, lbias = _bound_oneside(last_lA)
        return uA, ubias, lA, lbias
    
    def interval_propagate(self, h_U, h_L):
        r"""Function for forward propagation through a BoundedLinear layer.
        Args:
            h_U (tensor): The upper bound of the tensor input to this layer.

            h_L (tensor): The lower bound of the tensor input to this layer.

        Returns:
            upper (tensor): The upper bound of the output.
            lower (tensor): The lower bound of the output.
        """
        weight = self.weight
        bias = self.bias
        # Linf norm
        mid = (h_U + h_L) / 2.0
        diff = (h_U - h_L) / 2.0
        weight_abs = weight.abs()
        center = torch.addmm(bias, mid, weight.t())
        deviation = diff.matmul(weight_abs.t())
        upper = center + deviation
        lower = center - deviation
        return upper, lower
    

class BoundReLU(nn.ReLU):
    def __init__(self, inplace=False):
        super(BoundReLU, self).__init__(inplace)

    @staticmethod
    def convert(act_layer):
        r"""Convert a ReLU layer to BoundReLU layer

        Args:
            act_layer (nn.ReLU): The ReLU layer object to be converted.

        Returns:
            l (BoundReLU): The converted layer object.
        """
        l = BoundReLU(act_layer.inplace)
        return l
    
    def forward(self, x):
        r"""Overwrite the forward function to set the shape of the node
            during a forward pass
        """
        self.shape = x.shape
        return F.relu(x)
    
    def bound_backward(self, last_uA, last_lA, start_node=None, optimize=False):
        r"""Backward propagate through the ReLU layer.

        Args:
            last_uA (tensor): A (the coefficient matrix) that is backward-propagated to this layer
            (from the layers after this layer). It's exclusive for computing the upper bound.

            last_lA (tensor): A that is backward-propagated to this layer. It's exclusive for computing the lower bound.

            start_node (int): An integer indicating the start node of this backward propagation. It's used for selecting alphas.

            optimize (bool): Indicating whether we are optimizing parameters (alpha).

        Returns:
            uA (tensor): The new A for computing the upper bound after taking this layer into account.
            
            ubias (tensor): The bias (for upper bound) produced by this layer.
            
            lA( tensor): The new A for computing the lower bound after taking this layer into account.
            
            lbias (tensor): The bias (for lower bound) produced by this layer.
        """
        ## JC Refer to Lemma 2.1 to understand this math/code better
        # lb_r and ub_r are the bounds of input (pre-activation)
        # JC, because of these clamps, we must be dealing with the case u_j > 0 > l_j
        lb_r = self.lower_l.clamp(max=0)
        ub_r = self.upper_u.clamp(min=0)
        # avoid division by 0 when both lb_r and ub_r are 0
        ub_r = torch.max(ub_r, lb_r + 1e-8)
        # CROWN upper and lower linear bounds
        upper_d = ub_r / (ub_r - lb_r) # JC same as (u_j)/(U_j-l_j)
        upper_b = - lb_r * upper_d # JC same as -(u_j*l_j)/(u_j - l_j)
        upper_d = upper_d.unsqueeze(1) # JC adding a dimension of 1 at the second position such that upper_d is now a row vector
        if optimize:
            # selected_alpha has shape (2, dim_of_start_node, batch_size=1, dim_of_this_node)
            selected_alpha = self.alpha[start_node]
            if last_lA is not None:
                lb_lower_d = selected_alpha[0].permute(1, 0, 2) # JC permute swaps the depth and row vector here, but leaves column intact
            if last_uA is not None:
                ub_lower_d = selected_alpha[1].permute(1, 0, 2)
        else:
            # JC lb_lower_ = ub_lower_d = alpha where alpha will be either 0 or 1, while in 'optimize' alpha is adaptively selected; why is alpha here depending on upper_d > 0.5?
            # JC very small bug, but should be upper_d >= 0.5, not upper_d > 0.5
            lb_lower_d = ub_lower_d = (upper_d >= 0.5).float()   # CROWN lower bounds # JC element wise, if greater, then element becomes 1, else becomes 0
            # Save lower_d as initial alpha for optimization
            self.init_d = lb_lower_d.squeeze(1) # No need to save the extra dimension. # JC init_d is a row vector but we can just save it as a column vector since we know what its shape should be
        uA = lA = None
        ubias = lbias = 0
        # Choose upper or lower bounds based on the sign of last_A
        if last_uA is not None:
            pos_uA = last_uA.clamp(min=0) # JC retain shape of last_uA but only keep positive entries
            neg_uA = last_uA.clamp(max=0) # JC retain shape of last_uA but only keep negative entries
            uA = upper_d * pos_uA + ub_lower_d * neg_uA # JC element wise addition puts positive and negative entries together into new uA

            # JC for ReLU, only upper bounds have 'bias' terms that need to be taken into account
            mult_uA = pos_uA.view(last_uA.size(0), last_uA.size(1), -1)
            ubias = mult_uA.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)
        if last_lA is not None:
            neg_lA = last_lA.clamp(max=0)
            pos_lA = last_lA.clamp(min=0)
            lA = upper_d * neg_lA + lb_lower_d * pos_lA

            mult_lA = neg_lA.view(last_lA.size(0), last_lA.size(1), -1)
            lbias = mult_lA.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1) # JC as before, will get row vector (which is correct), but just save as a column vector
        return uA, ubias, lA, lbias
    
    def interval_propagate(self, h_U, h_L):
        # stored upper and lower bounds
        self.upper_u = h_U
        self.lower_l = h_L
        return F.relu(h_U), F.relu(h_L)
    
    def init_opt_parameters(self, start_nodes):
        r"""Initialize self.alpha with lower_d that are already saved at
        self.init_d during the initial CROWN backward propagation.

        Args:
            start_nodes (list): A list of start_node, each start_node is a dictionary
            {'idx', 'node'}. 'idx' is an integer indicating the position of the start node,
            while 'node' is the object of the start node.
        """
        self.alpha = OrderedDict()
        alpha_shape = self.shape
        alpha_init = self.init_d
        for start_node in start_nodes:
            ns = start_node['idx']
            size_s = start_node['node'].out_features
            self.alpha[ns] = torch.empty([2, size_s, *alpha_shape]) # The first dimension is alpha for lower and upper bound
            # Why 2? One for the upper bound and one for the lower bound.
            self.alpha[ns].data.copy_(alpha_init.data)
    
    def clip_alpha(self):
        r"""Clip alphas after an single update.
        Alpha should be bewteen 0 and 1.
        """
        for v in self.alpha.values():
            v.data = torch.clamp(v.data, 0, 1)


class BoundSequential(nn.Sequential):
    def __init__(self, *args):
        super(BoundSequential, self).__init__(*args)
        self.has_input_constraints = False # JC flag to check if the bounded model also has constraints on the input
        self.G = None
        self.h = None
    
    # Convert a Pytorch model to a model with bounds
    # @param seq_model Input pytorch model
    # @return Converted model
    @staticmethod
    def convert(seq_model):
        r"""Convert a Pytorch model to a model with bounds.
        Args:
            seq_model: An nn.Sequential module.
        
        Returns:
            The converted BoundSequential module.
        """
        layers = []
        for l in seq_model:
            if isinstance(l, nn.Linear):
                layers.append(BoundLinear.convert(l))
            elif isinstance(l, nn.ReLU):
                layers.append(BoundReLU.convert(l))
        return BoundSequential(*layers) # JC using star because BoundSequential is a child of nn.Sequential which receives *args or a tuple of layers, in essence, unpacking [layers] as multiple arguments
    
    def compute_bounds(self, x_U=None, x_L=None, upper=True, lower=True, optimize=False, use_input_constraints=False):
        r"""Main function for computing bounds.

        Args:
        Args:
            x_U (tensor): The upper bound of x.

            x_L (tensor): The lower bound of x.

            upper (bool): Whether we want upper bound.

            lower (bool): Whether we want lower bound.

            optimize (bool): Whether we optimize alpha.

        Returns:
            ub (tensor): The upper bound of the final output.

            lb (tensor): The lower bound of the final output.    
        """
        ub = lb = None
        if optimize:
            # alpha-CROWN
            if upper:
                ub, _ = self._get_optimized_bounds(x_L=x_L, x_U=x_U, upper=True, lower=False, use_input_constraints=use_input_constraints)
            if lower:
                _, lb = self._get_optimized_bounds(x_L=x_L, x_U=x_U, upper=False, lower=True, use_input_constraints=use_input_constraints)
        else:
            # CROWN
            ub, lb = self.full_backward_range(x_U=x_U, x_L=x_L, upper=upper, lower=lower)
        return ub, lb
    
    # Full CROWN bounds with all intermediate layer bounds computed by CROWN
    def full_backward_range(self, x_U=None, x_L=None, upper=True, lower=True, optimize=False, lambda_=None, G=None, h=None, use_input_constraints=False):
        r"""A full backward propagation. We are going to sequentially compute the 
        intermediate bounds for each linear layer followed by a ReLU layer. For each
        intermediate bound, we call self.backward_range() to do a backward propagation 
        starting from that layer.

        Args:
            x_U (tensor): The upper bound of x.

            x_L (tensor): The lower bound of x.

            upper (bool): Whether we want upper bound.

            lower (bool): Whether we want lower bound.

            optimize (bool): Whether we optimize alpha.

        Returns:
            ub (tensor): The upper bound of the final output.

            lb (tensor): The lower bound of the final output.    
        """
        modules = list(self._modules.values())
        # CROWN propagation for all layers
        for i in range(len(modules)):
            # We only need the bounds before a ReLU layer
            if isinstance(modules[i], BoundReLU):
                if isinstance(modules[i-1], BoundLinear):
                    # add a batch dimension
                    newC = torch.eye(modules[i-1].out_features).unsqueeze(0).repeat(x_U.shape[0], 1, 1) # JC take I matrix and make it R^{1 X out_features X out_featuers}, then add depth to it so there are shape[0] copies of the identity matrix i.e. newC exists in R^{x_U.shape[0] X .out_features X .out_features}
                    # JC verified in terminal but unsqueeze(0) above has no effect, simply make copies in the depth dimension
                    # JC this newC has a depth equivalent to the batch size which is simply x_U.shape[0], thus lower and upper bounds are computed simultaneuosly for every batch instance
                    # newC will always start as an identity matrix and will be transformed into upper and lower bounds during backward propogation
                    # Use CROWN to compute pre-activation bounds
                    # starting from layer i-1
                    ub, lb = self.backward_range(x_U=x_U, x_L=x_L, C=newC, upper=True, lower=True, start_node=i-1, optimize=optimize)

                    # JC try updating bounds in each layer
                    # if use_input_constraints:
                    #     ub, lb = self.backward_range(x_U=x_U, x_L=x_L, C=newC, upper=True, lower=True, start_node=i-1, optimize=optimize, lambda_=lambda_, G=G, h=h, use_input_constraints=use_input_constraints)
                    # else:
                    #     ub, lb = self.backward_range(x_U=x_U, x_L=x_L, C=newC, upper=True, lower=True, start_node=i-1, optimize=optimize)
                # Set pre-activation bounds for layer i (the ReLU layer) 
                # JC the bounds of the ReLu layer will be the bounds of the output before the activation (ReLu) function
                modules[i].upper_u = ub
                modules[i].lower_l = lb
        # Get the final layer bound
        
        # JC only use the Lagrange multipliers in the last step but this may be subject to change
        if use_input_constraints:
            return self.backward_range(x_U=x_U, x_L=x_L, C=torch.eye(modules[i].out_features).unsqueeze(0), upper=upper, lower=lower, start_node=i, optimize=optimize, lambda_=lambda_, G=G, h=h, use_input_constraints=use_input_constraints)
        else:
            return self.backward_range(x_U=x_U, x_L=x_L, C=torch.eye(modules[i].out_features).unsqueeze(0), upper=upper, lower=lower, start_node=i, optimize=optimize)

    def backward_range(self, x_U=None, x_L=None, C=None, upper=False, lower=True, start_node=None, optimize=False, lambda_=None, G=None, h=None, use_input_constraints=False):
        r"""The backward propagation starting from a given node. Can be used to compute intermediate bounds or the final bound.

        Args:
            x_U (tensor): The upper bound of x.

            x_L (tensor): The lower bound of x.

            C (tensor): The initial coefficient matrix. Can be used to represent the output constraints.
            But we don't have any constraints here. So it's just an identity matrix.

            upper (bool): Whether we want upper bound.

            lower (bool): Whether we want lower bound. 

            start_node (int): The start node of this propagation. It should be a linear layer.

            optimize (bool): Whether we optimize parameters.

        Returns:
            ub (tensor): The upper bound of the output of start_node.
            lb (tensor): The lower bound of the output of start_node.
        """
        if use_input_constraints:
            # JC ensure that the input constraints are available
            assert self.has_input_constraints, "Cannot use input constraints as they were never given"
            assert lambda_ is not None or G is not None or h is not None, "Input constraints but required values (lambda_, G, h) were not given"

        # start propagation from the last layer
        modules = list(self._modules.values()) if start_node is None else list(self._modules.values())[:start_node+1] # JC want all of the modules from the first layer to this current layer [0,start_node]
        upper_A = C if upper else None
        lower_A = C if lower else None
        upper_sum_b = lower_sum_b = x_U.new([0]) # JC creates a new tensor([0]) object that is the same datatype as x_U, thus upper and lower sum are initialized as 0 (dimensions will be appropiately changed after first addition)
        # JC starting from the current linear layer (and upper_A=lower_A?=IdentityMatrix), get the upper and lower bound of the input to this ith layer
        for i, module in enumerate(reversed(modules)):
            upper_A, upper_b, lower_A, lower_b = module.bound_backward(upper_A, lower_A, start_node, optimize)

            # JC collecting bias terms
            upper_sum_b = upper_b + upper_sum_b
            lower_sum_b = lower_b + lower_sum_b
        # sign = +1: upper bound, sign = -1: lower bound
        # JC this concrete bound actually passes in the lower and upper bounded input to get the actual bound on the output of this layer
        def _get_concrete_bound(A, sum_b, sign = -1,lambda_=None, G=None, h=None, use_input_constraints=False):
            if A is None:
                return None
            A = A.view(A.size(0), A.size(1), -1)
            # A has shape (batch, specification_size, flattened_input_size)
            x_ub = x_U.view(x_U.size(0), -1, 1)
            x_lb = x_L.view(x_L.size(0), -1, 1)
            center = (x_ub + x_lb) / 2.0
            diff = (x_ub - x_lb) / 2.0
            # JC .bmm performs a batch matrix multiplication where each batch will be multiplied by the input given to .bmm(...)
            # JC looking back at appendix D, center represents x^nom, and the second term represents (+/-)epsilon*||lower_A_{i,:}||_1 where the sign depends on what bound is being computed (neg. if lower bound)

            if use_input_constraints:

                # attempt 5 (using the actual minima and maxima)
                bound = sign * torch.abs((A.squeeze(0).T - G.T@lambda_).T@diff) + (A.squeeze(0).T - G.T@lambda_).T@center - sign * lambda_.T@h
                bound = bound.squeeze(-1) + sum_b

                # attempt 4 (should look similar to attempt 2)
                # bound = A.squeeze(0)@center + sign * A.squeeze(0).abs()@diff + lambda_.T@h - lambda_.T@G@center + sign * (lambda_.T@G).abs()@diff
                # bound = bound.squeeze(-1) + sum_b
                # attempt 3
                # bound = (A.squeeze(0) - G.T@lambda_).T@(center.squeeze(0)) + sign*(A.squeeze(0) - G.T@lambda_).abs().T@(diff.squeeze(0)) - sign * lambda_.T@h
                # bound = bound.squeeze(-1) + sum_b
                # print(f"Bound has shape {bound.shape}; A.shape {A.shape}; sum_b.shape {sum_b.shape}, center {center.shape}, x_lb {diff.shape}")
                # attempt 2
                # bound = A.bmm(center) + sign * A.abs().bmm(diff) + lambda_.T@h - lambda_.T@G@center + sign * (lambda_.T@G).abs()@diff
                # bound = bound.squeeze(-1) + sum_b 

                # attempt 1
                # bound = (A + lambda_.T@G).bmm(center).abs() + sign * (A + lambda_.T@G).abs().bmm(diff)
                # bound = bound.squeeze(-1) + sum_b + sign * lambda_.T@h

            else:
                # JC return the dual bound without dealing with Lagrange multipliers
                bound = A.bmm(center) + sign * A.abs().bmm(diff)
                bound = bound.squeeze(-1) + sum_b
            return bound
        lb = _get_concrete_bound(lower_A, lower_sum_b, sign=-1,lambda_=lambda_, G=G, h=h, use_input_constraints=use_input_constraints)
        ub = _get_concrete_bound(upper_A, upper_sum_b, sign=+1,lambda_=lambda_, G=G, h=h, use_input_constraints=use_input_constraints)
        if ub is None:
            ub = x_U.new([np.inf])
        if lb is None:
            lb = x_L.new([-np.inf])
        return ub, lb
    
    def _get_optimized_bounds(self, x_U=None, x_L=None, upper=False, lower=True, use_input_constraints=False):
        r"""The main function of alpha-CROWN.

        Args:
            x_U (tensor): The upper bound of x.

            x_L (tensor): The lower bound of x.

            upper (bool): Whether we want upper bound.

            lower (bool): Whether we want lower bound. 

        Returns:
            best_ret_u (tensor): Optimized upper bound of the final output.
            best_ret_l (tensor): Optimized lower bound of the final output.

        JC Notice that lower is true and upper is false. This is because upper alpha is typically optimal, thus the main optimization to be done is on the alpha slope for the lower bound of ReLU
        """
        # iteration = 1000 if use_input_constraints else 20# JC previously 20 but increasing number of iterations since we are not optimizing over more parameters
        iteration = 200
        modules = list(self._modules.values())
        self.init_alpha(x_U=x_U, x_L=x_L)
        alphas, parameters = [], []
        # JC obtains the parameters (includings alphas) to pass to Adam, references to the alphas, and a copy of the previous best alphas
        best_alphas = self._set_alpha(parameters, alphas, lr=1e-1)
        if use_input_constraints:
            # JC must extend the parameters to include lambdas if there are input constraints
            assert self.has_input_constraints, "No input constraints were given"
            assert self.G is not None, "G not given"
            assert self.h is not None, "h not given"
            lambda_rows = self.G.shape[0]
            lambda_ = torch.rand(lambda_rows, G.shape[1], requires_grad=True)
            # if lower is True:
            #     lambda_ = torch.tensor([[0.015097863377796374, 0.0, 0.0, 0.0, 0.01381309245806234]]).T
            # else:
            #     lambda_ = torch.tensor([[0.0, 0.0016282808635717343, 0.0, 0.0, 0.0]]).T
            parameters[0]['params'].extend([lambda_])
            loss_graph = np.array([i for i in range(iteration)], dtype=np.float32)
            loss_graph = np.vstack((loss_graph, np.zeros(iteration)))
            lambda_vals = np.array([i for i in range(iteration)], dtype=np.float32)
            lambda_vals = np.tile(lambda_vals, lambda_.shape[0] + 1).reshape(-1, iteration) 
        else:
            lambda_ = None
        # print(f"Using parameters: {parameters}")
        opt = optim.Adam(parameters, maximize=True if lower else False)
        # Create a weight vector to scale learning rate.
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 0.98)
        best_intermediate_bounds = {}
        final_intermediate_bounds = {}
        need_grad = True
        for i in range(iteration):
            if i == iteration - 1:
                # No grad update needed for the last iteration
                need_grad = False
            with torch.no_grad() if not need_grad else ExitStack():
                ub, lb = self.full_backward_range(x_U=x_U, x_L=x_L, upper=upper, lower=lower, optimize=True, lambda_=lambda_, G=self.G, h=self.h, use_input_constraints=use_input_constraints)
            if i == 0:
                # save results at the first iteration
                best_ret = []
                best_ret_l = _save_ret_first_time(lb, float('-inf'), best_ret)
                best_ret_u = _save_ret_first_time(ub, float('inf'), best_ret)
                for node_id, node in enumerate(modules):
                    if isinstance(node, BoundReLU):
                        new_intermediate = [node.lower_l.detach().clone(),
                                            node.upper_u.detach().clone()]
                        best_intermediate_bounds[node_id] = new_intermediate
                        # JC adding these for the final values
                        new_intermediate = [node.lower_l.detach().clone(),
                                            node.upper_u.detach().clone()]
                        final_intermediate_bounds[node_id] = new_intermediate        
            
            l = lb
            if lb is not None:
                l = torch.sum(lb)
            u = ub
            if ub is not None:
                u = torch.sum(ub)

            loss_ = l if lower else u
            loss = loss_.sum() # JC negative one because we want to maximize a lower bound but minimze an upper bound
            with torch.no_grad():
                # JC want to find the best lower and upper bound without calculating gradients 
                if (lower and lb[0, 0] > best_ret_l[0, 0]) or (not lower and ub[0, 0] < best_ret_u[0, 0]):
                    save_alphas = True
                else:
                    save_alphas = False

                save_alphas = False
    
                best_ret_l = torch.max(best_ret_l, lb)
                best_ret_u = torch.min(best_ret_u, ub)
                self._update_optimizable_activations(best_intermediate_bounds, best_alphas, save_alphas, final_intermediate_bounds)

            opt.zero_grad(set_to_none=True) # JC reset the gradients before we accumulate them using loss.backward()
            if i != iteration - 1:
                # We do not need to update parameters in the last step since the
                # best result already obtained
                loss.backward()
                opt.step()
            for _, node in enumerate(modules):
                if isinstance(node, BoundReLU):
                    with torch.no_grad():
                        node.clip_alpha()

            # JC also clip lambda
            if use_input_constraints:
                with torch.no_grad():
                    lambda_.data = torch.clip(lambda_.data, min=0.0)
                    lambda_vals[1:, i] = lambda_.data[:,0].detach().clone().numpy().flatten()
                    loss_graph[1, i] = loss.item()

            # print the lambda values for debugging
            scheduler.step()
        # Set all variables to their saved best values
        # JC NOTE that this assumes that the best alpha values will be 
        # found in the last iteration of gradient descent rather than some iteration in between
        # according to how they are updated in _update_optimizable_activations. best_intermediate_bounds
        # on the other hand is updated correctly
        with torch.no_grad():
            for idx, node in enumerate(modules):
                if isinstance(node, BoundReLU):
                    # Assigns a new dictionary
                    node.alpha = best_alphas[idx]
                    best_intermediate = best_intermediate_bounds[idx]
                    # best_intermediate = final_intermediate_bounds[idx]
                    node.lower_l.data = best_intermediate[0].data
                    node.upper_u.data = best_intermediate[1].data
        
        if (use_input_constraints):
            
            # # now perform gradient descent again but for lambdas
            # # JC must extend the parameters to include lambdas if there are input constraints
            # assert self.has_input_constraints, "No input constraints were given"
            # assert self.G is not None, "G not given"
            # assert self.h is not None, "h not given"
            # lambda_rows = self.G.shape[0]
            # lambda_ = torch.rand(lambda_rows, 2, requires_grad=True)
            # iteration = 200
            # loss_graph = np.array([i for i in range(iteration)], dtype=np.float32)
            # loss_graph = np.vstack((loss_graph, np.zeros(iteration)))
            # lambda_vals = np.array([i for i in range(iteration)], dtype=np.float32)
            # lambda_vals = np.tile(lambda_vals, lambda_.shape[0] + 1).reshape(-1, iteration) 

            # modules = list(self._modules.values())
            # n = len(modules) - 1
            
            # opt = optim.Adam([lambda_], lr=0.1, maximize=True if lower else False)
            # # Create a weight vector to scale learning rate.
            # # scheduler = optim.lr_scheduler.ExponentialLR(opt, 0.98)

            # x_L_np = x_L.numpy().flatten()
            # x_U_np = x_U.numpy().flatten()
            # if lower:
            #     print(f"x_L_np {x_L_np}")
            # else:
            #     print(f"x_U_np {x_U_np}")
            # if lower:
            #     a = np.array([[x_L_np[0], x_L_np[1], 0, 0],[0, 0, x_L_np[0], x_L_np[1]]])
            #     b = np.array([[-1.5552],[-1.2606]])
            # else:
            #     a = np.array([[x_U_np[0], x_U_np[1], 0, 0],[0, 0, x_U_np[0], x_U_np[1]]])
            #     b = np.array([[-0.0812],[1.7438]])
            
            # result = np.linalg.lstsq(a,b, rcond=None)
            # A_matrix = np.float32(result[0])
            # print(f"A_matrix {A_matrix.reshape(2,2)}, isclose {np.isclose(a@A_matrix, b)}")
            # A_matrix = torch.from_numpy(A_matrix).reshape(2,2)

            # x_ub = x_U.view(x_U.size(0), -1, 1)
            # x_lb = x_L.view(x_L.size(0), -1, 1)
            # center = (x_ub + x_lb) / 2.0
            # diff = (x_ub - x_lb) / 2.0
            # sign = -1 if lower else 1

            # if lower:
            #     print(f"Before adding constraints, best_lb = {best_ret_l}")
            # else:
            #     print(f"Before adding constraints, best_ub = {best_ret_u}")

            # for i in range(iteration):
            #     curr_ub, curr_lb = self.backward_range(x_U=x_U, x_L=x_L, C=torch.eye(modules[n].out_features).unsqueeze(0), upper=upper, lower=lower, start_node=n, optimize=True, lambda_=lambda_, G=G, h=h, use_input_constraints=True)
            #     # curr_ub, curr_lb = self.full_backward_range(x_U=x_U, x_L=x_L, upper=upper, lower=lower, optimize=True, lambda_=lambda_, G=self.G, h=self.h, use_input_constraints=use_input_constraints)
            #     # y = (A_matrix - G.T@lambda_).T@(center.squeeze(0)) + sign*(A_matrix - G.T@lambda_).abs().T@(diff.squeeze(0)) - sign * lambda_.T@h

            #     with torch.no_grad():
            #         best_ret_l = torch.max(best_ret_l, curr_lb)
            #         best_ret_u = torch.min(best_ret_u, curr_ub)

            #     # with torch.no_grad():
            #     #     best_ret_l = torch.max(best_ret_l, y)
            #     #     best_ret_u = torch.min(best_ret_u, y)

            #     if lower:
            #         y = curr_lb.sum()
            #         # y = curr_lb.squeeze()[0]
            #     else:
            #         y = curr_ub.sum()
            #         # y = curr_ub.squeeze()[0]
            #     # y = y.sum()

            #     opt.zero_grad(set_to_none=True)
            #     y.backward()
            #     opt.step()
            #     # scheduler.step()

            #     with torch.no_grad():
            #         lambda_.data = torch.clip(lambda_.data, min=0.0)
            #         lambda_vals[1:, i] = lambda_.data[:,0].detach().clone().numpy().flatten()
            #         loss_graph[1, i] = y.item()

            # without gradient descent
            # curr_ub, curr_lb = self.backward_range(x_U=x_U, x_L=x_L, C=torch.eye(modules[n].out_features).unsqueeze(0), upper=upper, lower=lower, start_node=n, optimize=True, lambda_=lambda_, G=G, h=h, use_input_constraints=True)
            # best_ret_l = torch.max(curr_lb, curr_lb)
            # best_ret_u = torch.min(curr_ub, curr_ub)
            # if lower:
            #     best_ret_l = (A_matrix - G.T@lambda_).T@(center.squeeze(0)) + sign*(A_matrix - G.T@lambda_).abs().T@(diff.squeeze(0)) - sign * lambda_.T@h
            #     best_ret_l.squeeze(-1)
            #     best_ret_u = float('inf')
            #     print(f"a crown with input best_ret_l {best_ret_l}")
            # else:
            #     best_ret_l = float('-inf')
            #     best_ret_u = (A_matrix - G.T@lambda_).T@(center.squeeze(0)) + sign*(A_matrix - G.T@lambda_).abs().T@(diff.squeeze(0)) - sign * lambda_.T@h
            #     best_ret_u.squeeze(-1)
            #     print(f"a crown with input best_ret_u {best_ret_u}")

            # display losses 
            fig = plt.figure(figsize=(18,12))
            num_subs = 1 + lambda_.shape[0]
            num_rows = ceil(num_subs / 2)
            for i in range(num_subs):
                plt.subplot(num_rows, 2, i + 1)
                
                if i == 0:
                    plt.title(f"Loss Function for {'Lower Bound' if upper is False else 'Upper Bound'} w/ Final {loss_graph[1,-1]}")
                    plt.ylabel(f"y")
                    plt.plot(loss_graph[0,:], loss_graph[1,:])
                else:
                    plt.title(f"Lambda {i} w.r.t. steps")
                    plt.ylabel(f"Lambda {i}")
                    plt.plot(lambda_vals[0,:], lambda_vals[i,:])

                plt.xlabel("Steps")

            plt.show()
            print(f"Lambda values: {lambda_.data}")
        elif (not use_input_constraints):
            modules = list(self._modules.values())
            i = len(modules) - 1
            test_ub, test_lb = self.backward_range(x_U=x_U, x_L=x_L, C=torch.eye(modules[i].out_features).unsqueeze(0), upper=upper, lower=lower, start_node=i, optimize=True)
            if lower:
                print(f"Lower Alpha Crown without Input Constraints | lb: {test_lb}")
            else:
                print(f"Upper Alpha Crown without Input Constraints | ub: {test_ub}")

        return best_ret_u, best_ret_l
    
    def add_input_constraints(self, G, h):
        print(f"Adding input constraints with G.shape {G.shape} and h.shape{h.shape}")
        self.G = G
        self.h = h
        self.has_input_constraints = True

    def init_alpha(self, x_U=None, x_L=None):
        r"""Initialize alphas and intermediate bounds for alpha-CROWN
        Contains a full CROWN method.

        Args:
            x_U (tensor): The upper bound of x.

            x_L (tensor): The lower bound of x.

        Returns:
            lb (tensor): Lower CROWN bound.

            ub (tensor): Upper CROWN bound.

            init_intermediate_bounds (dictionary): Intermediate bounds obtained 
            by initial CROWN.
        """
        # Do a forward pass to set perturbed nodes
        self(x_U)
        # Do a CROWN to init all intermediate layer bounds and alpha
        ub, lb = self.full_backward_range(x_U, x_L)
        modules = list(self._modules.values())
        # Also collect the initial intermediate bounds
        init_intermediate_bounds = {}
        for i, module in enumerate(modules):
            if isinstance(module, BoundReLU):
                start_nodes = self.get_alpha_crown_start_nodes(i)
                module.init_opt_parameters(start_nodes)
                init_intermediate_bounds[i-1] = [module.lower_l, module.upper_u]
        return lb, ub, init_intermediate_bounds
    
    def _set_alpha(self, parameters, alphas, lr):
        r"""Collect alphas from all the ReLU layers and gather them
        into "parameters" for optimization. Also construct best_alphas
        to keep tracking the values of alphas.

        Args:
            parameters (list): An empty list, to gather all alphas for optimization.

            alphas (list): An empty list, to gather all values of alphas.

            lr (float): Learning rate, for optimization.

        best_alphas (OrderDict): An OrderDict object to collect the value of alpha.
        """
        modules = list(self._modules.values())
        for i, node in enumerate(modules):
            if isinstance(node, BoundReLU):
                alphas.extend(list(node.alpha.values()))
        # Alpha has shape (2, output_shape, batch_dim, node_shape)
        parameters.append({'params': alphas, 'lr': lr, 'batch_dim': 2})
        # best_alpha is a dictionary of dictionary. Each key is the alpha variable
        # for one actiation layer, and each value is a dictionary contains all
        # activation layers after that layer as keys.
        best_alphas = OrderedDict()
        for i, node in enumerate(modules):
            if isinstance(node, BoundReLU):
                best_alphas[i] = {}
                for alpha_node in node.alpha:
                    # JC first get the initial alphas and save this as best_alphas without adding
                    # the gradient tree
                    best_alphas[i][alpha_node] = node.alpha[alpha_node].detach().clone()
                    # JC now require that each alpha's gradient gets saved for the subsequent 
                    # passes through the Adam optimizer
                    node.alpha[alpha_node].requires_grad_()
        return best_alphas

    # For a given node, return the list of indices of its "start_nodes"
    # A "start_node" of a given node is a node from which a backward propagation uses the given node,
    # so we will store a set of alpha for that "start_node" with the given node.
    def get_alpha_crown_start_nodes(self, node_id):
        modules = list(self._modules.values())
        start_nodes = []
        for i in range(node_id, len(modules)):
            if isinstance(modules[i], BoundLinear):
                start_nodes.append({'idx': i, 'node': modules[i]})
        return start_nodes
    
    # Update bounds and alpha of optimizable activations
    def _update_optimizable_activations(self, best_intermediate_bounds, best_alphas, save_alphas=False, final_intermediate_bounds=None):
        modules = list(self._modules.values())
        for i, node in enumerate(modules):
            if isinstance(node, BoundReLU):
                if save_alphas:
                    for alpha_m in node.alpha:
                            best_alphas[i][alpha_m] = node.alpha[alpha_m].detach().clone()

                    if final_intermediate_bounds is not None:
                        # JC save the higher (tighter) lower bound as the best
                        final_intermediate_bounds[i][0] = node.lower_l.detach().clone()
                        # JC save the lower (tighter) upper bound as the best
                        final_intermediate_bounds[i][1] = node.upper_u.detach().clone()
                    
                # JC save the higher (tighter) lower bound as the best
                best_intermediate_bounds[i][0] = torch.max(
                    best_intermediate_bounds[i][0],
                    node.lower_l
                )
                # JC save the lower (tighter) upper bound as the best
                best_intermediate_bounds[i][1] = torch.min(
                    best_intermediate_bounds[i][1],
                    node.upper_u
                )

                for alpha_m in node.alpha:
                    best_alphas[i][alpha_m] = node.alpha[alpha_m].detach().clone()
                
        
    
# Save results at the first iteration to best_ret.
def _save_ret_first_time(bounds, fill_value, best_ret):
    if bounds is not None:
        best_bounds = torch.full_like(bounds, fill_value=fill_value, dtype=torch.float32)
    else:
        best_bounds = None
    if bounds is not None:
        best_ret.append(bounds.detach().clone())
    else:
        best_ret.append(None)
    return best_bounds



if __name__ == '__main__':
    model = Model()
    # model.load_state_dict(torch.load('very_simple_model.pth'))
    # model.load_state_dict(torch.load('very_stupid_model.pth'))
    model.load_state_dict(torch.load('large_model.pth'))

    input_width = model.model[0].in_features
    output_width = model.model[-1].out_features

    torch.manual_seed(14)
    batch_size = 1
    x = torch.rand(batch_size, input_width)
    print("x: {} shape {} | output: {}".format(x, x.shape, model(x)))
    eps = 6
    x_u = x + eps
    x_l = x - eps
    # x_u = torch.tensor([[4.0, 6.0]])
    # x_l = torch.tensor([[-2.0, -5.0]])
    print(f"x_u shape {x_u.shape}, x_l shape {x_l.shape}")

    print("%%%%%%%%%%%%%%%%%%%%%%%% CROWN %%%%%%%%%%%%%%%%%%%%%%%%%%")
    boundedmodel = BoundSequential.convert(model.model)
    ub, lb = boundedmodel.compute_bounds(x_U=x_u, x_L=x_l, upper=True, lower=True)
    for i in range(batch_size):
        for j in range(output_width):
            print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
                j=j, i=i, l=lb[i][j].item(), u=ub[i][j].item()))
        print('---------------------------------------------------------')
    print()
    
    print("%%%%%%%%%%%%%%%%%%%%% alpha-CROWN %%%%%%%%%%%%%%%%%%%%%%%")
    boundedmodel = BoundSequential.convert(model.model)
    ub, lb = boundedmodel.compute_bounds(x_U=x_u, x_L=x_l, upper=True, lower=True, optimize=True)
    for i in range(batch_size):
        for j in range(output_width):
            print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
                j=j, i=i, l=lb[i][j].item(), u=ub[i][j].item()))
        print('---------------------------------------------------------')
    print()

    # JC this is where the results of input constrained will be printed
    print("%%%%%%%%%%%%% alpha-CROWN with constraints %%%%%%%%%%%%%%")
    # JC initializing the input constraints
    # G = torch.tensor([[5/3, 1], [-1/8, 1]])
    # h = torch.tensor([[-25/3], [-19/4]])
    G = torch.tensor([[2/9, -1], [6/5, -1], [5/3, 1], [-1/8, 1], [-5, -1]])
    h = torch.tensor([[-46/9], [-6], [-25/3], [-19/4], [-26]])
    G = torch.tensor([[1/3, -1],[3, 1],[-1/3, 1], [-3, -1]])
    h = torch.tensor([[-5/3],[-5],[-5/3],[-5]])
    boundedmodel = BoundSequential.convert(model.model)
    boundedmodel.add_input_constraints(G,h) # JC add constraints on the input
    ub, lb = boundedmodel.compute_bounds(x_U=x_u, x_L=x_l, upper=True, lower=True, optimize=True, use_input_constraints=True)
    for i in range(batch_size):
        for j in range(output_width):
            print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
                j=j, i=i, l=lb[i][j].item(), u=ub[i][j].item()))
        print('---------------------------------------------------------')
    print()

    print("%%%%%%%%%%%%%%%%%%%%% auto-LiRPA %%%%%%%%%%%%%%%%%%%%%%%%")
    image = x
    lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device,
                                bound_opts={'sparse_intermediate_bounds': False,
                                            'sparse_features_alpha': False})
    norm = float("inf")
    ptb = PerturbationLpNorm(norm=norm, eps=eps)
    image = BoundedTensor(image, ptb) # JC This bounded tensor represents the pertubation set or contour C

    for method in ['backward (CROWN)', 'CROWN-Optimized']:
        print('Bounding method:', method)
        if 'Optimized' in method:
            # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can
            # increase verbosity to see per-iteration loss values.
            lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}})
        lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0])
        for i in range(batch_size):
            for j in range(output_width):
                print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
                    j=j, i=i, l=lb[i][j].item(), u=ub[i][j].item()))
            print('---------------------------------------------------------')
        print()
