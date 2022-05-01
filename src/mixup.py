import warnings
warnings.filterwarnings("ignore")
import torch, faiss
import pdb
import cvxpy as cp
import numpy as np
import copy
import sys
import random


def pos_mixup(tensor, num_id):
    batch_size = tensor.shape[0]
    num_pos = int(batch_size/num_id)
    for i in range(0, batch_size, num_pos):
        if num_pos == 6:
            alpha = np.random.rand()
            fake_1 = alpha*tensor[i,:] + (1.-alpha)*tensor[i+1,:]
            fake_1 = torch.unsqueeze(fake_1, 0)

            alpha = np.random.rand()
            fake_2 = alpha*tensor[i+1,:] + (1.-alpha)*tensor[i+2,:]
            fake_2 = torch.unsqueeze(fake_2, 0)

            alpha = np.random.rand()
            fake_3 = alpha*tensor[i+2,:] + (1.-alpha)*tensor[i+3,:]
            fake_3 = torch.unsqueeze(fake_3, 0)

            alpha = np.random.rand()
            fake_4 = alpha*tensor[i+3,:] + (1.-alpha)*tensor[i+4,:]
            fake_4 = torch.unsqueeze(fake_4, 0)

            alpha = np.random.rand()
            fake_5 = alpha*tensor[i+4,:] + (1.-alpha)*tensor[i+5,:]
            fake_5 = torch.unsqueeze(fake_5, 0)

            alpha = np.random.rand()
            fake_6 = alpha*tensor[i,:] + (1.-alpha)*tensor[i+2,:]
            fake_6 = torch.unsqueeze(fake_6, 0)

            alpha = np.random.rand()
            fake_7 = alpha*tensor[i,:] + (1.-alpha)*tensor[i+3,:]
            fake_7 = torch.unsqueeze(fake_7, 0)

            alpha = np.random.rand()
            fake_8 = alpha*tensor[i,:] + (1.-alpha)*tensor[i+4,:]
            fake_8 = torch.unsqueeze(fake_8, 0)

            alpha = np.random.rand()
            fake_9 = alpha*tensor[i,:] + (1.-alpha)*tensor[i+5,:]
            fake_9 = torch.unsqueeze(fake_9, 0)

            alpha = np.random.rand()
            fake_10 = alpha*tensor[i+1,:] + (1.-alpha)*tensor[i+3,:]
            fake_10 = torch.unsqueeze(fake_10, 0)

            alpha = np.random.rand()
            fake_11 = alpha*tensor[i+1,:] + (1.-alpha)*tensor[i+4,:]
            fake_11 = torch.unsqueeze(fake_11, 0)

            alpha = np.random.rand()
            fake_12 = alpha*tensor[i+1,:] + (1.-alpha)*tensor[i+5,:]
            fake_12 = torch.unsqueeze(fake_12, 0)

            alpha = np.random.rand()
            fake_13 = alpha*tensor[i+2,:] + (1.-alpha)*tensor[i+4,:]
            fake_13 = torch.unsqueeze(fake_13, 0)

            alpha = np.random.rand()
            fake_14 = alpha*tensor[i+2,:] + (1.-alpha)*tensor[i+5,:]
            fake_14 = torch.unsqueeze(fake_14, 0)

            alpha = np.random.rand()
            fake_15 = alpha*tensor[i+3,:] + (1.-alpha)*tensor[i+5,:]
            fake_15 = torch.unsqueeze(fake_15, 0)

            if i == 0:
                tensor_fake = torch.cat((fake_1, fake_2, fake_3, fake_4, fake_5, fake_6, fake_7, fake_8, fake_9, fake_10, fake_11, fake_12, fake_13, fake_14, fake_15), dim=0)
            else:
                tensor_fake = torch.cat((tensor_fake, fake_1, fake_2, fake_3, fake_4, fake_5, fake_6, fake_7, fake_8, fake_9, fake_10, fake_11, fake_12, fake_13, fake_14, fake_15), dim=0)

        if num_pos == 4:
            alpha = np.random.rand()
            fake_1 = alpha*tensor[i,:] + (1.-alpha)*tensor[i+1,:]
            fake_1 = torch.unsqueeze(fake_1, 0)

            alpha = np.random.rand()
            fake_2 = alpha*tensor[i+1,:] + (1.-alpha)*tensor[i+2,:]
            fake_2 = torch.unsqueeze(fake_2, 0)

            alpha = np.random.rand()
            fake_3 = alpha*tensor[i+2,:] + (1.-alpha)*tensor[i+3,:]
            fake_3 = torch.unsqueeze(fake_3, 0)

            alpha = np.random.rand()
            fake_4 = alpha*tensor[i,:] + (1.-alpha)*tensor[i+3,:]
            fake_4 = torch.unsqueeze(fake_4, 0)

            alpha = np.random.rand()
            fake_5 = alpha*tensor[i,:] + (1.-alpha)*tensor[i+2,:]
            fake_5 = torch.unsqueeze(fake_5, 0)

            alpha = np.random.rand()
            fake_6 = alpha*tensor[i+1,:] + (1.-alpha)*tensor[i+3,:]
            fake_6 = torch.unsqueeze(fake_6, 0)

            if i == 0:
                tensor_fake = torch.cat((fake_1, fake_2, fake_3, fake_4, fake_5, fake_6), dim=0)
            else:
                tensor_fake = torch.cat((tensor_fake, fake_1, fake_2, fake_3, fake_4, fake_5, fake_6), dim=0)
        elif num_pos == 3:
            alpha = np.random.rand()
            fake_1 = alpha*tensor[i,:] + (1.-alpha)*tensor[i+1,:]
            fake_1 = torch.unsqueeze(fake_1, 0)

            alpha = np.random.rand()
            fake_2 = alpha*tensor[i+1,:] + (1.-alpha)*tensor[i+2,:]
            fake_2 = torch.unsqueeze(fake_2, 0)

            alpha = np.random.rand()
            fake_3 = alpha*tensor[i,:] + (1.-alpha)*tensor[i+2,:]
            fake_3 = torch.unsqueeze(fake_3, 0)

            if i == 0:
                tensor_fake = torch.cat((fake_1, fake_2, fake_3), dim=0)
            else:
                tensor_fake = torch.cat((tensor_fake, fake_1, fake_2, fake_3), dim=0)
    ind = num_pos
    if num_pos == 6: num_fakes = 15
    elif num_pos == 4: num_fakes = 6
    elif num_pos == 3: num_fakes = 3
    for i in range(0, tensor_fake.shape[0], num_fakes):
        tensor = torch.cat((tensor[:ind,:], tensor_fake[i:i+num_fakes,:], tensor[ind:,:]), dim=0)
        ind += num_pos + num_fakes
    return tensor

