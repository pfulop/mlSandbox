import torch
from torch import nn


class DCGAND(nn.Module):
    def __init__(self, n_extra_layers):
        super(DCGAND, self).__init__()

        main = nn.Sequential()

        channels = 3
        out_channels = 64
        size = 64

        self.conv_block(main, f'start-conv', channels, out_channels, 4, 2, 1)

        size /= 2

        for i in range(n_extra_layers):
            self.conv_block(main, f'{i}-conv', out_channels, out_channels, 3, 1, 1)

        while size > 4:
            self.conv_block(main, 'pyramid', out_channels, out_channels * 2, 4, 2, 1)
            out_channels, size = out_channels * 2, size / 2

        main.add_module(f'final-conv', nn.Conv2d(out_channels, 1, 4, 1, 0, bias=False))
        self.main = main

    def conv_block(self, main, name, inf, outf, size, stride, padding):
        main.add_module(f'{name}-{inf}.{outf}.conv', nn.Conv2d(inf, outf, size, stride, padding, bias=False))
        main.add_module(f'{name}-{outf}.batchnorm', nn.BatchNorm2d(outf))
        main.add_module(f'{name}-{outf}.relu', nn.LeakyReLU(0.2, inplace=True))

    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor):
            gpu_ids = range(1)
        output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        output = output.mean(0)
        return output.view(1)


class DCGANG(nn.Module):
    def deconv_block(self, main, name, inf, of, a, b, c):
        main.add_module(f'{name}-{inf}.{of}.convt', nn.ConvTranspose2d(inf, of, a, b, c, bias=False))
        main.add_module(f'{name}-{of}.batchnorm', nn.BatchNorm2d(of))
        main.add_module(f'{name}-{of}.relu', nn.ReLU(inplace=True))

    def __init__(self, n_extra_layers=0):
        super(DCGANG, self).__init__()
        size = 64
        isize = size
        out_channels = 64

        out_channels, tisize = out_channels // 2, 4
        while tisize != size: out_channels *= 2; tisize *= 2

        main = nn.Sequential()
        self.deconv_block(main, 'initial', 100, out_channels, 4, 1, 0)

        size = 4
        while size < isize // 2:
            self.deconv_block(main, 'pyramid', out_channels, out_channels // 2, 4, 2, 1)
            out_channels //= 2;
            size *= 2

        for t in range(n_extra_layers):
            self.deconv_block(main, f'extra-{t}', out_channels, out_channels, 3, 1, 1)

        main.add_module(f'final.convt', nn.ConvTranspose2d(out_channels, 3, 4, 2, 1, bias=False))
        main.add_module(f'final.tanh', nn.Tanh())
        self.main = main

    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) :
            gpu_ids = range(1)
        return nn.parallel.data_parallel(self.main, input, gpu_ids)

