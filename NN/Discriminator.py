import torch.nn as nn
from torch.nn.utils import weight_norm


class DiscLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        model = nn.ModuleDict()

        model["0"] = nn.Sequential(
            weight_norm(nn.Conv1d(128, n_features, 7)),
            nn.LeakyReLU(0.2),
        )

        downsampling = 4
        to_n_features = n_features

        for i in range(4):
            from_n_features = to_n_features
            to_n_features = to_n_features * 2

            model["%d" % (i+1)] = nn.Sequential(
                weight_norm(nn.Conv1d(from_n_features, to_n_features, 5,
                                      stride=downsampling, groups=from_n_features // downsampling, padding=4))
            )

        model["5"] = nn.Sequential(
            weight_norm(nn.Conv1d(to_n_features, to_n_features, 1, padding=2)),
            nn.LeakyReLU(0.2)
        )

        model["6"] = nn.Sequential(
            weight_norm(nn.Conv1d(to_n_features, 128, 1)),
            nn.LeakyReLU(0.2)
        )

        self.model = model

    def forward(self, x):
        results = []
        for i, D_layer in self.model.items():
            x = D_layer(x)
            results.append(x)
        return results


class Discriminator(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.model = nn.ModuleDict()
        self.downsampling = nn.AvgPool1d(4, 2)

        for i in range(3):
            self.model["D_layer%d" % i] = DiscLayer(n_features)

    def forward(self, x):
        results = []
        for i, d in self.model.items():
            results.append(d(x))
            x = self.downsampling(x)
        return results
