import torch
from ..helpers.utils import gumbel_sigmoid
from .masked_nn import MaskedNN

class MaskedModel(torch.nn.Module):

    def __init__(self, model_type, n_samples, n_nodes, pns_mask, hidden_layers,
                 hidden_dim, l1_graph_penalty, seed, device) -> None:
        super(MaskedModel, self).__init__()
        self.model_type = model_type
        self.n_samples = n_samples
        self.n_nodes = n_nodes
        self.pns_mask = pns_mask
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.l1_graph_penalty = l1_graph_penalty
        self.seed = seed
        self.device = device

        if self.model_type == 'nn':
            self.masked_model = MaskedNN(mask=self.pns_mask,
                                         n_samples=self.n_samples,
                                         hidden_layers=self.hidden_layers,
                                         hidden_dim=self.hidden_dim,
                                         device=self.device
                                         )
        else:
            raise TypeError(f"The argument `model_type` must be "
                            f"['nn'], but got {self.model_type}.")
        torch.manual_seed(self.seed)
        w = torch.nn.init.uniform_(torch.Tensor(self.n_nodes, self.n_nodes),
                                   a=-1e-10, b=1e-10)
        self.w = torch.nn.Parameter(w.to(device=self.device))

    def forward(self, x, rho, alpha, temperature) -> tuple:

        w_prime = self._preprocess_graph(self.w, tau=temperature,
                                         seed=self.seed)
        w_prime = self.pns_mask * w_prime
        mse_loss = self._get_mse_loss(x, w_prime)
        h = (torch.trace(torch.matrix_exp(w_prime * w_prime)) - self.n_nodes)
        loss = (0.5 / self.n_samples * mse_loss
                + self.l1_graph_penalty * torch.linalg.norm(w_prime, ord=1)
                + alpha * h
                + 0.5 * rho * h * h)

        return loss, h, self.w

    def _preprocess_graph(self, w, tau, seed=0) -> torch.Tensor:
    #w为10*10
        w_prob = gumbel_sigmoid(w, temperature=tau, seed=seed,
                                device=self.device)
        w_prob = (1. - torch.eye(w.shape[0], device=self.device)) * w_prob
    #去对角线
        return w_prob

    def _get_mse_loss(self, x, w_prime):

        mse_loss = 0
        for i in range(self.n_nodes):
            # Get possible PNS parents and also remove diagonal element
            pns_parents = torch.where(self.pns_mask[:, i] == 1)[0]
            possible_parents = [int(j) for j in pns_parents if j != i]
            if len(possible_parents) == 0:    # Root node, don't have to build NN in this case
                continue
            curr_x = x[:, possible_parents]    # Features for current node
            curr_y = x[:, i]   # Label for current node
            curr_w = w_prime[possible_parents, i]   # Mask for current node

            curr_masked_x = curr_x * curr_w  # Broadcasting
            curr_y_pred = self.masked_model(curr_masked_x,
                                            choice=str(i), num=len(possible_parents))  # Use masked features to predict value of current node
            nosie = torch.nn.init.uniform_(torch.Tensor(self.n_samples, len(possible_parents)), a=-0.05, b=0.05)
            curr_nosie_pred = self.masked_model(nosie * curr_w, choice=str(i), num=len(possible_parents))
            curr_y_nosie_pred = curr_y_pred + curr_nosie_pred
            mse_loss = mse_loss + torch.sum(torch.square(curr_y_nosie_pred.squeeze() - curr_y))

        return mse_loss