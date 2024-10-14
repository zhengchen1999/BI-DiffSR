import torch
import lpips
from torch import nn

def disabled_train(self: nn.Module) -> nn.Module:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def frozen_module(module: nn.Module) -> None:
    module.eval()
    module.train = disabled_train
    for p in module.parameters():
        p.requires_grad = False

class LPIPS:
    def __init__(self, net: str) -> None:
        self.model = lpips.LPIPS(net=net)
        frozen_module(self.model)
    
    @torch.no_grad()
    def __call__(self, img1: torch.Tensor, img2: torch.Tensor, normalize: bool, boundarypixels=0) -> torch.Tensor:
        """
        Compute LPIPS.
        
        Args:
            img1 (torch.Tensor): The first image (NCHW, RGB, [-1, 1]). Specify `normalize` if input 
                image is range in [0, 1].
            img2 (torch.Tensor): The second image (NCHW, RGB, [-1, 1]). Specify `normalize` if input 
                image is range in [0, 1].
            normalize (bool): If specified, the input images will be normalized from [0, 1] to [-1, 1].
            
        Returns:
            lpips_values (torch.Tensor): The lpips scores of this batch.
        """

        b, c, h, w = img1.shape
        img1 = img1[:, :, :h-h%boundarypixels, :w-w%boundarypixels]
        # img1 = img1[:,:, boundarypixels:-boundarypixels,boundarypixels:-boundarypixels]
        b, c, h, w = img2.shape
        img2 = img2[:, :, :h-h%boundarypixels, :w-w%boundarypixels]
        # img2 = img2[:,:, boundarypixels:-boundarypixels,boundarypixels:-boundarypixels]

        return self.model(img1, img2, normalize=normalize)
    
    def to(self, device: str) -> "LPIPS":
        self.model.to(device)
        return self
