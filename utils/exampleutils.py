import torch
import numpy as np




def merge_worker_matrices(num_parts, save_dir, base_matrix_path):
    # TODO make the pathing pretty

    base_matrix = torch.load(base_matrix_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    examples_losses = torch.zeros(base_matrix.shape, device=device)

    for i in range(num_parts):
        # load the base matrix in cuda:0
        worker_matrix = torch.load(save_dir + f'/Rank {i}- examples_losses.pt').to('cuda:0')
        examples_losses += worker_matrix

    return examples_losses