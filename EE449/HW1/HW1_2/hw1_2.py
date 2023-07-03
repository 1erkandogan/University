import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import os
import torch
from torchvision.utils import make_grid

def part2Plots(out, nmax=64, save_dir='', filename=''):
    out = torch.tensor(out).reshape(-1,1,25,25)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid((out.detach()[:nmax]), nrow=8).permute(1, 2, 0))
    fig.savefig(os.path.join(save_dir, filename + '.png'))

def my_conv2d(input, kernel):
    # Get input and kernel shapes
    batch_size, input_channels, input_height, input_width = input.shape
    output_channels, input_channels, filter_height, filter_width = kernel.shape
    
    # Compute output tensor shape
    output_height = input_height - filter_height + 1
    output_width = input_width - filter_width + 1
    
    # Initialize output tensor with zeros
    output = np.zeros((batch_size, output_channels, output_height, output_width))
    
    # Loop over batch, output channels, and output spatial dimensions
    for b in range(batch_size):
        for c_out in range(output_channels):
            for i in range(output_height):
                for j in range(output_width):
                    # Compute dot product between input patch and kernel patch
                    output[b, c_out, i, j] = np.sum(
                        input[b, :, i:i+filter_height, j:j+filter_width] * kernel[c_out]
                    )
                    
    return output



def hw1_2():
    #input shape: [batch_size, input_Channels, input_height, input_width]
    input = np.load('samples_2.npy')
    #input shape: [output_channels, input_Channels, filter_height, filter_width]
    kernel = np.load('kernel.npy')
    out = my_conv2d(input, kernel)
    part2Plots(out, save_dir = '.', filename = 'output')
    
hw1_2()