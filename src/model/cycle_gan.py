import torch.nn


class CycleGAN(torch.nn.Module):
    def __init__(self, disc, gen):
        """
        Parameters:
            disc - discriminator
            gen - generator
        """
        super(CycleGAN, self).__init__()
        self.disc = disc
        self.gen = gen

    def forward(self, *input):
        pass



