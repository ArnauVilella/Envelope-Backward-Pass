import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import cvxpy as cp

class EnvelopeFunction(autograd.Function):
    @staticmethod
    def forward(ctx, variables, parameters, objective, inequalities, equalities, cvxpy_opts, *batch_params):
        cp_inequalities = [ineq(*variables, *parameters) <= 0 for ineq in inequalities]
        cp_equalities = [eq(*variables, *parameters) == 0 for eq in equalities]
        problem = cp.Problem(
            cp.Minimize(objective(*variables, *parameters)),
            cp_inequalities + cp_equalities
        )

        outputs = []
        gradients = []

        for batch in range(batch_params[0].shape[0]):
            with torch.no_grad():
                params = [p[batch] for p in batch_params]
                for i, p in enumerate(parameters):
                    p.value = params[i].double().numpy()
                problem.solve(**cvxpy_opts)

                z = [torch.tensor(v.value, dtype=params[0].dtype, device=params[0].device) for v in variables]
                lam = [torch.tensor(c.dual_value, dtype=params[0].dtype, device=params[0].device)
                       for c in cp_inequalities]
                nu = [torch.tensor(c.dual_value, dtype=params[0].dtype, device=params[0].device)
                      for c in cp_equalities]
                optimal_val = torch.tensor(problem.value, dtype=params[0].dtype, device=params[0].device)

            with torch.enable_grad():
                params = [p[batch].detach().requires_grad_(True) for p in batch_params]

                g = [ineq(*z, *params) for ineq in inequalities]
                h = [eq(*z, *params) for eq in equalities]
                L = (objective(*z, *params) +
                     sum((u * v).sum() for u, v in zip(lam, g)) +
                     sum((u * v).sum() for u, v in zip(nu, h)))

                dproblem = autograd.grad(L, params, create_graph=True)

            outputs.append(optimal_val)
            gradients.append(dproblem)

        ctx.save_for_backward(*batch_params)
        ctx.gradients = gradients
        return torch.stack(outputs)

    @staticmethod
    def backward(ctx, grad_output):
        batch_params = ctx.saved_tensors
        gradients = ctx.gradients

        batch_size = len(gradients)
        grad_inputs = [None] * len(batch_params)
        for i in range(len(batch_params)):
            grad_inputs[i] = torch.stack([gradients[batch][i] * grad_output[batch] for batch in range(batch_size)])

        return (None, None, None, None, None, None, *grad_inputs)


class EnvelopeLayer(nn.Module):
    def __init__(self, variables, parameters, objective, inequalities, equalities, **cvxpy_opts):
        super().__init__()
        self.variables = variables
        self.parameters = parameters
        self.objective = objective
        self.inequalities = inequalities
        self.equalities = equalities
        self.cvxpy_opts = cvxpy_opts

    def forward(self, *batch_params):
        return EnvelopeFunction.apply(
            self.variables,
            self.parameters,
            self.objective,
            self.inequalities,
            self.equalities,
            self.cvxpy_opts,
            *batch_params
        )