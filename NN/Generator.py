import torch.nn as nn
from torch.nn.utils import weight_norm
 

class Generator(nn.Module):
    def __init__(self, batch_size, n_features):
        super().__init__()

        upsampling_rate = [8, 2]

        mult = 16

        model = [
            # nn.ReflectionPad1d(3),
            weight_norm(nn.Conv1d(128,128, 5))
        ]

        for i in range(2):
            u_r = upsampling_rate[i]

            for ii in range(2):
                model += [
                    nn.LeakyReLU(0.2),
                    weight_norm(nn.ConvTranspose1d(
                        128, 128, u_r, u_r, u_r // 2 + u_r % 2, u_r % 2)),
                ]

                for residual_layer in range(3):
                    model += [
                        nn.Sequential(
                            nn.LeakyReLU(0.2),
                            weight_norm(nn.Conv1d(128, 128, kernel_size=3,
                                                  dilation=3 ** residual_layer)),
                            nn.LeakyReLU(0.2),
                            weight_norm(
                                nn.Conv1d(128, 128, kernel_size=1)),

                        )
                    ]
                mult //= 2

        model += [
            nn.LeakyReLU(0.2),
            # nn.ReflectionPad1d(5),
            weight_norm(nn.Conv1d(128, 128, stride=251,
                                  kernel_size=1)), 
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(128, 128,
                                  kernel_size=1, padding=0)),                                 
            nn.ReflectionPad1d((1,1)),    
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
