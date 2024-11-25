import torch
from torchviz import make_dot
from unet_main import Unet

pseudo_data = torch.randn(1, 1, 256, 256)

model = Unet(1, 1)

y = model(pseudo_data)
image = make_dot(y, params = dict(model.named_parameters()))
image.format = "png"
image.render("UNet")