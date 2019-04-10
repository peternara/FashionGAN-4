from torch.nn import Module, Conv2d, LeakyReLU, GroupNorm, Sequential, Linear


class Discriminator(Module):
    def __init__(self, num_input_channels, num_out_channels=64, num_layers=8, kernel_size=4, num_classification=10):
        """

        :param num_input_channels:
        :param num_out_channels:
        :param num_layers:
        :param kernel_size:

        Notes:
        use leaky relu to allow small negative gradients to pass to the generator. https://sthalles.github.io/advanced_gans/
        Use group norm because it performs better than batch norm for small batches
        """
        super(Discriminator, self).__init__()
        pad= 1
        seq = [Conv2d(num_input_channels, num_out_channels, kernel_size=kernel_size, stride=2, padding=1, bias=True),
               LeakyReLU(0.2, True)]
        for i in range(1, int(num_layers/2)):
            seq += [
                Conv2d(num_out_channels * i, num_out_channels * (i+1), kernel_size=kernel_size, stride=2, padding=pad, bias=True),
                GroupNorm(num_groups=8, num_channels=num_out_channels* (i+1)),
                LeakyReLU(0.2, True),
                Conv2d(num_out_channels * (i+1), num_out_channels * (i + 1), kernel_size=kernel_size, stride=1, padding=pad,
                       bias=True),
                GroupNorm(num_groups=8, num_channels=num_out_channels * (i + 1)),
                LeakyReLU(0.2, True)
            ]
        seq += [
            Conv2d(num_out_channels * int(num_layers/2), num_out_channels * int(num_layers/ 4), kernel_size=kernel_size,
                   stride=1, padding=pad, bias=True),
            GroupNorm(num_groups=8, num_channels=num_out_channels * int(num_layers/ 4)),
            LeakyReLU(0.2, True),
            Conv2d(num_out_channels * int(num_layers / 4), num_out_channels, kernel_size=kernel_size,
                   stride=1, padding=pad, bias=True)
        ]
        self.conv = Sequential(*seq)
        self.gan_layer = Linear(num_out_channels, 2)
        self.classification_layer = Linear(num_out_channels, num_classification)

    def forward(self, *input):
        res = self.conv(input)
        return self.gan_layer(res)

    def classify(self, *input):
        res = self.conv(input)
        return self.classification_layer(res)

