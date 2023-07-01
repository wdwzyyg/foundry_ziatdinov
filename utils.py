#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 00:30:02 2019

@author: Maxim Ziatdinov
"""

import torch

class Hook():
    """
    Returns the input and output of a
    layer during forward/backward pass

    see https://www.kaggle.com/sironghuang/
        understanding-pytorch-hooks/notebook
    """
    def __init__(self, module, backward=False):
        """
        Args:
            module: torch modul(single layer or sequential block)
            backward (bool): replace forward_hook with backward_hook
        """
        if backward is False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

def mock_forward(model, dims=(1, 64, 64)):
    '''Passes a dummy variable throuh a network'''
    x = torch.randn(1, dims[0], dims[1], dims[2])
    out = model(x)
    return out
