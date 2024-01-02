import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from model import Model
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from collections import OrderedDict
from contextlib import ExitStack


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
            self.alpha[ns] = torch.empty([2, size_s, *alpha_shape]) # The first diminsion of alpha_shape is batch size.
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
        return BoundSequential(*layers) # using star because BoundSequential is a child of nn.Sequential which receives *args or a tuple of layers, in essence, unpacking [layers] as multiple arguments
    
    def compute_bounds(self, x_U=None, x_L=None, upper=True, lower=True, optimize=False):
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
                ub, _ = self._get_optimized_bounds(x_L=x_L, x_U=x_U, upper=True, lower=False)
            if lower:
                _, lb = self._get_optimized_bounds(x_L=x_L, x_U=x_U, upper=False, lower=True)
        else:
            # CROWN
            ub, lb = self.full_backward_range(x_U=x_U, x_L=x_L, upper=upper, lower=lower)
        return ub, lb
    
    # Full CROWN bounds with all intermediate layer bounds computed by CROWN
    def full_backward_range(self, x_U=None, x_L=None, upper=True, lower=True, optimize=False):
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
                # Set pre-activation bounds for layer i (the ReLU layer) 
                # JC the bounds of the ReLu layer will be the bounds of the output before the activation (ReLu) function
                modules[i].upper_u = ub
                modules[i].lower_l = lb
        # Get the final layer bound
        return self.backward_range(x_U=x_U, x_L=x_L, C=torch.eye(modules[i].out_features).unsqueeze(0), upper=upper, lower=lower, start_node=i, optimize=optimize)

    def backward_range(self, x_U=None, x_L=None, C=None, upper=False, lower=True, start_node=None, optimize=False):
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
        def _get_concrete_bound(A, sum_b, sign = -1):
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
            bound = A.bmm(center) + sign * A.abs().bmm(diff)
            bound = bound.squeeze(-1) + sum_b
            return bound
        lb = _get_concrete_bound(lower_A, lower_sum_b, sign=-1)
        ub = _get_concrete_bound(upper_A, upper_sum_b, sign=+1)
        if ub is None:
            ub = x_U.new([np.inf])
        if lb is None:
            lb = x_L.new([-np.inf])
        return ub, lb
    
    def _get_optimized_bounds(self, x_U=None, x_L=None, upper=False, lower=True):
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
        modules = list(self._modules.values())
        self.init_alpha(x_U=x_U, x_L=x_L)
        alphas, parameters = [], []
        # JC obtains the parameters (includings alphas) to pass to Adam, references to the alphas, and a copy of the previous best alphas
        best_alphas = self._set_alpha(parameters, alphas, lr=1e-1)
        opt = optim.Adam(parameters)
        # Create a weight vector to scale learning rate.
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 0.98)
        iteration = 20
        best_intermediate_bounds = {}
        need_grad = True
        for i in range(iteration):
            if i == iteration - 1:
                # No grad update needed for the last iteration
                need_grad = False
            with torch.no_grad() if not need_grad else ExitStack():
                ub, lb = self.full_backward_range(x_U=x_U, x_L=x_L, upper=upper, lower=lower, optimize=True)
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
            l = lb
            if lb is not None:
                l = torch.sum(lb)
            u = ub
            if ub is not None:
                u = torch.sum(ub)        
            loss_ = l if lower else -u
            loss = (-1 * loss_).sum() # JC negative one because we want to maximize a lower bound but minimze an upper bound
            with torch.no_grad():
                # JC want to find the best lower and upper bound without calculating gradients 
                best_ret_l = torch.max(best_ret_l, lb)
                best_ret_u = torch.min(best_ret_u, ub)
                self._update_optimizable_activations(best_intermediate_bounds, best_alphas)
            opt.zero_grad(set_to_none=True) # JC reset the gradients before we accumulate them using loss.backward()
            if i != iteration - 1:
                # We do not need to update parameters in the last step since the
                # best result already obtained
                loss.backward()
                opt.step()
            for i, node in enumerate(modules):
                if isinstance(node, BoundReLU):
                    node.clip_alpha()
            scheduler.step()
        # Set all variables to their saved best values
        with torch.no_grad():
            for idx, node in enumerate(modules):
                if isinstance(node, BoundReLU):
                    # Assigns a new dictionary
                    node.alpha = best_alphas[idx]
                    best_intermediate = best_intermediate_bounds[idx]
                    node.lower_l.data = best_intermediate[0].data
                    node.upper_u.data = best_intermediate[1].data
        return best_ret_u, best_ret_l

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
                    best_alphas[i][alpha_node] = node.alpha[alpha_node].detach().clone()
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
    def _update_optimizable_activations(self, best_intermediate_bounds, best_alphas):
        modules = list(self._modules.values())
        for i, node in enumerate(modules):
            if isinstance(node, BoundReLU):
                best_intermediate_bounds[i][0] = torch.max(
                    best_intermediate_bounds[i][0],
                    node.lower_l
                )
                best_intermediate_bounds[i][1] = torch.min(
                    best_intermediate_bounds[i][1],
                    node.upper_u
                )
                for alpha_m in node.alpha:
                    best_alphas[i][alpha_m] = node.alpha[alpha_m]
        
    
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
    # torch.save(model.state_dict(), 'model_verysimple.pth')
    # model.load_state_dict(torch.load('model.pth'))

    input_width = model.model[0].in_features
    output_width = model.model[-1].out_features

    # torch.manual_seed(14)
    batch_size = 5
    x = torch.rand(batch_size, input_width)
    print("output: {}".format(model(x)))
    eps = 1
    x_u = x + eps
    x_l = x - eps

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