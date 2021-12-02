import torch
from torch import nn, einsum
from einops import rearrange


class LambdaLayer(nn.Module):
    def __init__(
        self,
        in_channels, # number of input channels
        out_channels, # numbr of output channels
        query_depth = 16, # number of channels for the keys
        intra_depth = 1, # number of nieghboring slices, intra-depth dimension
        receptive_kernel = 3, # local context kernel size
        temporal_kernel = 1,
        heads = 1, # number of query heads
        num_slices = 1
    ):
        super().__init__()
        self.dim_in = in_channels
        self.dim_out = out_channels
        
        self.q_depth = query_depth
        self.intra_depth = intra_depth

        assert (out_channels % heads) == 0, 'out_channels must be divisible by number of heads for multi-head query.'
        self.v_depth = out_channels // heads
        self.heads = heads
        
        self.num_slices = num_slices
        
        self.receptive_kernel = receptive_kernel
        self.temporal_kernel = temporal_kernel

        self.to_q = nn.Sequential(
            nn.Conv2d(in_channels, query_depth * heads, kernel_size=1, bias=False),
            nn.BatchNorm2d(query_depth * heads)
        )
        self.to_k = nn.Sequential(
            nn.Conv2d(in_channels, query_depth * intra_depth, kernel_size=1, bias=False),
        )
        self.to_v = nn.Sequential(
            nn.Conv2d(in_channels, self.v_depth * intra_depth, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.v_depth * intra_depth)
        )

        assert (receptive_kernel % 2) == 1, 'Receptive kernel size should be odd.'
        self.pos_conv = nn.Conv3d(intra_depth, query_depth, (1, receptive_kernel, receptive_kernel),
                                  padding = (0, receptive_kernel // 2, receptive_kernel // 2))
        
        if temporal_kernel >= 3:
            assert (temporal_kernel <= num_slices)
            assert (temporal_kernel % 2) == 1, 'Temporal kernel size should be odd.'
            self.temp_conv = nn.Conv2d(intra_depth, query_depth, (1, temporal_kernel),
                                      padding = (0, temporal_kernel // 2))

    def forward(self, input):
        batch, channel, height, width = *input.shape, 

        q = self.to_q(input)
        k = self.to_k(input)
        v = self.to_v(input)

        q = rearrange(q, 'b (h k) hh ww -> b h k (hh ww)', h = self.heads)
        k = rearrange(k, 'b (u k) hh ww -> b u k (hh ww)', u = self.intra_depth)
        v = rearrange(v, 'b (u v) hh ww -> b u v (hh ww)', u = self.intra_depth)

        k = k.softmax(dim=-1)

        λc = einsum('b u k m, b u v m -> b k v', k, v)
        Yc = einsum('b h k n, b k v -> b h v n', q, λc)

        v_p = rearrange(v, 'b u v (hh ww) -> b u v hh ww', hh = height, ww = width)
        λp = self.pos_conv(v_p)
        Yp = einsum('b h k n, b k v n -> b h v n', q, λp.flatten(3))


        if self.temporal_kernel >= 3:
            v_t = rearrange(v, '(g t) u v p -> (g p) u v t', t = self.num_slices)
            λt = self.temp_conv(v_t)
            λt = rearrange(λt, '(g p) k v t -> (g t) k v p', p = height*width)
            Yt = einsum('b h k n, b k v n -> b h v n', q, λt)
            Y = Yc + Yp + Yt
        else:
            Y = Yc + Yp
            
        out = rearrange(Y, 'b h v (hh ww) -> b (h v) hh ww', hh = height, ww = width)
        return out
    
class LambdaBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float, temporal_kernel: int = 1, num_slices: int = 1):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
            temporal_kernel: 
            num_slices: should be larger or equal to tk
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            LambdaLayer(in_chans, out_chans, temporal_kernel=temporal_kernel, num_slices=num_slices, heads=max(1, out_chans//32), intra_depth=4),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            LambdaLayer(out_chans, out_chans, temporal_kernel=temporal_kernel, num_slices=num_slices, heads=max(1, out_chans//32), intra_depth=4),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)