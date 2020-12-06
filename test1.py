import models
import torch

models.ONNX_EXPORT = True
model = models.Darknet(model_cfg, img_size=img_size)
dummy_input = torch.randn(batch_size, 3, img_size, img_size)
torch.onnx.export(model, dummy_input, 'file.onnx')
