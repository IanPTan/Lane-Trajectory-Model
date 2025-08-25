"""
This is a script to export the TwinLiteNetPlus model as an onnx model to be run with hte onnx runtime. This file should be placed in the root of the TwinLiteNetPlus repository to be ran properly.
"""


import torch
from argparse import ArgumentParser
import time
import numpy as np

from model.model import TwinLiteNetPlus



parser = ArgumentParser()
parser.add_argument('--weight', type=str, default='pretrained/large.pth', help='model.pth path(s)')
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--config', type=str, choices=["nano", "small", "medium", "large"], help='Model configuration')
parser.add_argument('--save-dir', type=str, default='model.onnx', help='directory to save model')
opt = parser.parse_args()



model = TwinLiteNetPlus(opt)
model.load_state_dict(torch.load(opt.weight, map_location=torch.device("cpu")))
dummy_input = torch.zeros((1, 3, 384, 640), device="cpu")

print("Exporting...")
torch.onnx.export(model,
              dummy_input,
              opt.save_dir,
              export_params=True,
              opset_version=10,
              do_constant_folding=True,
              input_names=['image'],
              output_names=['mask1', 'mask2'],
              dynamic_axes={'image': {0: 'batch_size'},
                            'mask1': {0: 'batch_size'},
                            'mask2': {0: 'batch_size'}})

print(f"Successfully exported {opt.config}.onnx!")
