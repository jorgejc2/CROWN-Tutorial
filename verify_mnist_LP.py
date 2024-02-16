"""
This script using Linear Programming to verify model performance. Relies on the triangle relaxation for ReLU.
Gives bounds only, so if the model passes verification on a given image, performance is guaranteed; however, if the model
    fails verification on a given image, it not actually misclassify any images perturbed in plus or minus epsilon. That
    is to say, false negatives are impossible, but false positives are possible.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
# from model import Model
# from simple_model import Model
# from stupid_model import Model
from large_model import Model
from gurobipy import GRB, quicksum, max_
from tqdm import trange
from copy import deepcopy

from gurobi_utils import get_gurobi_model


class MNISTModelVerifier:
    """
    This class uses Linear Programming to verify performance of an MNISTModel.
    """

    def __init__(self, model=None):
        """
        Saves or creates an MNISTModel for later verification.
        :param model: The MNISTModel instance to verify. If None, will attempt to load from the default configuration
                        (2 layers of 100 hidden units, located at mnist_model.pt)
        """
        if model is None:
            raise Exception
        
        self.net = model

    def verify(self, dataset, epsilon, silence_gurobi=True, silence_tqdm=False, silence_print=True):
        """
        Verify performance of self.model on dataset with perturbation level epsilon, return the indices of verified images.
        :param dataset: An MNIST dataset--iterable of (image, label) tensors. Should support indexing or return value
                        will not be meaningful.
        :param epsilon: Perturbation level. Each pixel may be independently perturbed within [pixel - epsilon, pixel +
                        epsilon], clamped at [0, 1], creating an L-infinity ball with the image at the center.
        :param silence_gurobi: Whether to silence output from the gurobi LP solver. Default True.
        :param silence_tqdm: Whether to silence the progress bars. Default False.
        :param silence_print: Whether to silence print statements. Default True.
        :return: List of verified images from the dataset.
        """
        loop = trange(len(dataset), disable=silence_tqdm, leave=False)
        loop.set_description("Verification")

        # The main loop over the dataset
        verified = [index for index in loop if self.verify_one(*dataset[index], epsilon, silence_gurobi=silence_gurobi)]

        if not silence_print:
            print(f"Verified {len(verified) / len(dataset):.2%} of the dataset at epsilon={epsilon}.", end=" ")
            if len(verified) >= len(dataset) / 2:
                print("Failed: ", list(filter(lambda i: i not in verified, range(len(dataset)))))
            else:
                print("Verified: ", verified)

        return verified

    def verify_one(self, image, label, epsilon, save_solution_filename=None, silence_gurobi=True, with_input_constraints=False):
        """
        Verify performance on a single instance.
        :param image: Image tensor from the MNIST dataset. Should have 28x28 pixels.
        :param label: The corresponding label.
        :param epsilon: Perturbation level. Each pixel may be independently perturbed within [pixel - epsilon, pixel +
                        epsilon], clamped at [0, 1]
        :param save_solution_filename: If provided, will save all variables from the solver to this location.
        :param silence_gurobi: Whether to silence output from the gurobi LP solver.
        :return: True if the model's performance was verified on this image, else False.
        """
        lb, ub, ineq_pi = self._calculate_bound(image, label, epsilon, save_solution_filename=save_solution_filename,
                                      silence_gurobi=silence_gurobi, with_input_constraints=with_input_constraints)
        return lb, ub, ineq_pi

    def _calculate_bound(self, image, label, epsilon, save_solution_filename=None, silence_gurobi=True, with_input_constraints=False):
        m = get_gurobi_model(silence_gurobi)

        # Add the input neuron variables and constraints
        image = image.flatten()
        n_in = image.shape[0]
        print(f"Input size | {n_in}")
        # lb = (image - epsilon * np.ones(n_in)).clip(min=0, max=1)
        # ub = (image + epsilon * np.ones(n_in)).clip(min=0, max=1)
        lb = (image - epsilon * np.ones(n_in))
        ub = (image + epsilon * np.ones(n_in))
        print(f"Image: \n{image}\nlb: \n{lb}\nub: \n{ub}")
        activations = np.array(list(m.addVars(n_in, lb=lb, ub=ub, name="x").values()))
        # JC if manually adding the bounds on the input
        # if not with_input_constraints:
        activations = m.addMVar(shape=(n_in), name='x', lb=lb, ub=ub)
        og_activations = activations
        # else:
        #     activations = m.addMVar(shape=(n_in), name='x', lb=np.array([-2.0, -5.0]), ub=np.array([4.0, 6.0]))
        pre_activation = None  # We assume the network doesn't start with a ReLU

        upper_dual_variables = {}
        lower_dual_variables = {}
        dual_layer_count = 0

        # JC finally adding constraints to the input
        # print(m.getVarByName('x'))
        # x0 = m.getVarByName('x[0]')
        # x1 = m.getVarByName('x[1]')
        # x_in = [x0, x1]
        ineq_constraints = []
        if with_input_constraints:
            G = np.array([[2/9, -1], [6/5, -1], [5/3, 1], [-1/8, 1], [-5, -1]])
            h = np.array([[-46/9], [-6], [-25/3], [-19/4], [-26]])
            G = np.array([[1/3, -1],[3, 1],[-1/3, 1], [-3, -1]])
            h = np.array([[-5/3],[-5],[-5/3],[-5]])
            for i in range(G.shape[0]):
                ineq_constraints.append(m.addConstr(quicksum(G[i,j] * og_activations[j] for j in range(G.shape[1])) >= h[i], name=f'input_constr_{i}'))

        # JC get the modules per layer
        for i, module in enumerate(self.net.model.children()):
            print(f"Module {i + 1}: {module}")
        modules = list(self.net.model.children())

        # JC original modules
        # modules = self.net.net

        # Go through sequential net and tighten bounds using LP at each layer
        last_constrs = None
        for index, module in enumerate(modules):

            # JC printing more information
            print(f"index: {index} | Currently bounding layer {module}")

            if isinstance(module, torch.nn.Linear):
                # JC print the weight values
                print(f"Weights of linear layer {i}: \n {module.weight}\n {module.bias}")

                # For linear layers, we extract parameters from the model and create exact constraints between
                # activations from the previous layer and pre-activation values for the next layer.
                parameters = dict(module.named_parameters())
                weight = parameters["weight"].detach().numpy()
                bias = parameters["bias"].detach().numpy()
                n_out, n_in = weight.shape

                pre_activation = np.array(list(m.addVars(n_out, lb=float("-inf"), name=f"z{index}").values()))
                last_constrs = m.addConstrs((
                    pre_activation[i] == quicksum([weight[i][j] * activations[j] for j in range(n_in)]) + bias[i]
                    for i in range(n_out)),
                    f"{index}.Linear",
                )

                n_in = n_out
            elif isinstance(module, torch.nn.ReLU):
                # For ReLU layers, we use the LP solver to create bounds on pre_activation variables, and use the triangle
                # relaxation to turn those bounds into bounds on activation variables. This is where inexactness is
                # introduced, meaning that failed verification may be a false postive.

                # Compute bounds
                lb, ub = np.zeros_like(pre_activation), np.zeros_like(pre_activation)
                for idx, neuron in enumerate(pre_activation):
                    m.setObjective(neuron, sense=GRB.MINIMIZE)
                    m.optimize()
                    # relaxed = m.relax()
                    # relaxed.optimize()
                    lb[idx] = m.ObjVal
                    # lb[idx] = relaxed.ObjVal

                    # now add the lower bound dual variables to the dictionary for comparison
                    # curr_lb_duals = [const.pi for const in last_constrs]
                    # lower_dual_variables[dual_layer_count] = curr_lb_duals

                    m.setObjective(neuron, sense=GRB.MAXIMIZE)
                    m.optimize()
                    # relaxed = m.relax()
                    # relaxed.optimize()
                    ub[idx] = m.ObjVal
                    # ub[idx] = relaxed.ObjVal

                    # now add the upper bound dual variables to the dictionary for comparison
                    # curr_ub_duals = [const.pi[0] for const in last_constrs]
                    # upper_dual_variables[dual_layer_count] = curr_ub_duals

                    dual_layer_count += 1

                # JC by default, addVars sets the lower bound to be 0.0
                activations = np.array(list(m.addVars(n_in, name=f"a{index}").values()))

                for i in range(n_in):
                    if lb[i] >= 0:
                        m.addConstr(activations[i] == pre_activation[i], name=f"{index}.ReLU.activation_{i}")
                    elif ub[i] <= 0:
                        m.addConstr(activations[i] == 0, name=f"{index}.ReLU.activation_{i}")
                    else:  # lb[i] < 0 < ub[i]:
                        # Triangle relaxation for ReLU: a is greater than 0, greater than z, and less than the line connecting
                        # the lower bound's activation and the upper bound's activation.
                        # The first constraint is handled by default--see docs for addVars
                        # m.addConstr(activations[i] >= 0, name=f"{index}.ReLU.activation_{i}_triangle_1")
                        m.addConstr(activations[i] >= pre_activation[i], name=f"{index}.ReLU.activation_{i}_triangle_2")
                        m.addConstr(
                            activations[i] <= (pre_activation[i] - lb[i]) * ub[i] / (ub[i] - lb[i]),
                            name=f"{index}.ReLU.activation_{i}_triangle_3",
                        )
            elif isinstance(module, torch.nn.Flatten):
                pass
            else:
                raise TypeError(f"Verifier not equipped to handle layer of type {type(module)}")

        # Minimize the gap between the correct output logit and the largest incorrect logit.
        output_neurons = pre_activation  # We assume no ReLU on output neurons

        # JC before this would find the robustness to the worst pertubation
        # max_incorrect_logit = m.addVar(lb=float("-inf"), name="max_incorrect_logit")
        # m.addConstr(max_incorrect_logit == max_([var for i, var in enumerate(output_neurons) if i != label]),
        #             name="max_incorrect_logit")
        # m.setObjective(output_neurons[label] - max_incorrect_logit, GRB.MINIMIZE)
        # ineq_constraints = []
        # if with_input_constraints:
        #     G = np.array([[2/9, -1], [6/5, -1], [5/3, 1], [-1/8, 1], [-5, -1]])
        #     h = np.array([[-46/9], [-6], [-25/3], [-19/4], [-26]])
        #     G = np.array([[1/3, -1],[3, 1],[-1/3, 1], [-3, -1]])
        #     h = np.array([[-5/3],[-5],[-5/3],[-5]])
        #     for i in range(G.shape[0]):
        #         ineq_constraints.append(m.addConstr(quicksum(G[i,j] * og_activations[j] for j in range(G.shape[1])) >= h[i], name=f'input_constr_{i}'))
        
        # JC this now find the bounds on each logit
        lb = torch.zeros((1, len(output_neurons)), dtype=torch.float32)
        ub = torch.zeros((1, len(output_neurons)), dtype=torch.float32)
        ineq_pi = []
        print(f"Number of inequality constraints: {len(ineq_constraints)}")
        for i in range(len(output_neurons)):
            m.setObjective(output_neurons[i], GRB.MINIMIZE)
            m.optimize()
            # relaxed = m.relax()
            # relaxed.optimize()
            # append dual variables for the lower bound

            # ineq_pi.append([const.pi[0] for const in ineq_constraints])
            # ineq_pi[-1].extend([f"lb: {i}",])
            min_val = m.ObjVal
            # min_val = relaxed.ObjVal
            lb[0,i] = min_val
            print(f"lb[0,{i}] = {min_val}")
            min_vars = ""
            for v in m.getVars():
                min_vars += '%s: %g\n' % (v.VarName, v.x)


            m.setObjective(output_neurons[i], GRB.MAXIMIZE)
            m.optimize()
            # relaxed = m.relax()
            # relaxed.optimize()
            # append dual variables for the upper bound
            # ineq_pi.append([const.pi[0] for const in ineq_constraints])
            # ineq_pi[-1].extend([f"ub: {i}",])
            # max_val = m.ObjVal
            max_val = m.ObjVal
            ub[0,i] = max_val
            print(f"ub[0,{i}] = {max_val}")
            max_vars = ""
            for v in m.getVars():
                max_vars += '%s: %g\n' % (v.VarName, v.x)

            print(f"i: {i} | {min_val} <= f(x) <= {max_val}")      
            # print("Min Vars")
            # print(min_vars)
            # print("Max Vars")
            # print(max_vars)

        print(f"Final Output: \nUpper duals\n {upper_dual_variables}\nLower duals\n {lower_dual_variables}")

        # if save_solution_filename:
        #     try:
        #         with open(save_solution_filename, "w") as f:
        #             for v in m.getVars():
        #                 f.write(f"{v.varName} {v.x:.6}\n")
        #     except FileNotFoundError:
        #         raise ValueError(f"Could not write solution to {save_solution_filename}")

        # return m.ObjVal
            
        return lb, ub, ineq_pi


if __name__ == "__main__":
    # Example usage:

    # # 1. Verify a portion of the validation dataset
    # verifier = MNISTModelVerifier()
    # dataset = [validation_dataset[i] for i in range(10)]
    # verifier.verify(dataset=dataset, epsilon=0.05, silence_print=False)

    # # 2. Verify a specific image, saving the solution to a file
    # verified = verifier.verify_one(*dataset[0], epsilon=0.05, save_solution_filename="LP_solution.txt")

    # JC testing out MNIST verifier on a very simple model
    model = Model()
    # model.load_state_dict(torch.load('very_simple_model.pth'))
    # model.load_state_dict(torch.load("very_stupid_model.pth"))
    model.load_state_dict(torch.load('large_model.pth'))

    print("Printing structure of 'very_stupid_model.pth'")
    for name, param in model.named_parameters():
        print(f"Layer: {name}, Size: {param.size()}")

    verifier = MNISTModelVerifier(model)
    
    torch.manual_seed(14)
    x = torch.rand(1, 2)
    epsilon = 6
    print(x)

    input_width = model.model[0].in_features
    output_width = model.model[-1].out_features

    lb_a, ub_a, ineq_pi_a = verifier.verify_one(x, label=0, epsilon=epsilon, save_solution_filename="LP_solution_to_very_simple_model.txt", with_input_constraints=False)
    lb_b, ub_b, ineq_pi_b = verifier.verify_one(x, label=0, epsilon=epsilon, save_solution_filename="LP_solution_to_very_simple_model.txt", with_input_constraints=True)
    print(f"lb_a shape: {len(lb_a)}, ub_a shape: {len(ub_a)}, ineq_pi_a shape {len(ineq_pi_a)}")
    print(f"lb_b shape: {len(lb_b)}, ub_a shape: {len(ub_b)}, ineq_pi_a shape {len(ineq_pi_b)}")
    print("%%%%%%%%%%%%%%%%%%%%%%%% GUROBI %%%%%%%%%%%%%%%%%%%%%%%%%%")
    for i in range(1):
        for j in range(output_width):
            print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
                j=j, i=i, l=lb_a[i][j].item(), u=ub_a[i][j].item()))
        print('---------------------------------------------------------')
    print()
    
    print("%%%%%%%%%%%%%%%%%%%%%%%% GUROBI w/ input constraints %%%%%%%%%%%%%%%%%%%%%%%%%%")
    for i in range(1):
        for j in range(output_width):
            print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
                j=j, i=i, l=lb_b[i][j].item(), u=ub_b[i][j].item()))
        print('---------------------------------------------------------')
    print()
    for i, pi in enumerate(ineq_pi_b):
        print(pi)
