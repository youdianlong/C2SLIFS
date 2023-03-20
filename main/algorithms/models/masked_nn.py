import torch
import torch.nn as nn

class MaskedNN(nn.Module):

    def __init__(self, mask, n_samples, hidden_layers, hidden_dim,
                 device=None) -> None:
        super(MaskedNN, self).__init__()
        self.mask = mask  # use mask to determine input dimension
        self.n_samples = n_samples
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self._init_nn()

    def forward(self, x, choice, num) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
            possible parents
        choice: str of int
            current sub-note y

        Returns
        -------
        output: torch.Tensor
            shape = (n,)
        """

        b = torch.rand(self.n_samples, 1)
        output = self.nets[choice](x.float()) + b

        return output

    def _init_nn(self):
        """ Initialize net for each node"""

        md = {}
        for i in range(self.mask.shape[0]):
            pns_parents = torch.where(self.mask[:, i] == 1)[0]
            first_input_dim = len([int(j) for j in pns_parents if j != i])
            if first_input_dim == 0:    # Root node, don't have to build NN in this case
                continue
            reg_nn = []
            for j in range(self.hidden_layers):
                input_dim = self.hidden_dim
                if j == 0:
                    input_dim = first_input_dim
                func = nn.Sequential(
                    nn.Linear(in_features=input_dim,
                          out_features=self.hidden_dim).to(device=self.device),
                    nn.LeakyReLU(negative_slope=0.05).to(device=self.device)
                )
                reg_nn.append(func)
            output_layer = nn.Linear(in_features=self.hidden_dim,
                                     out_features=1).to(device=self.device)
            reg_nn.append(output_layer)
            reg_nn = nn.Sequential(*reg_nn)

            md[str(i)] = reg_nn
        self.nets = nn.ModuleDict(md)