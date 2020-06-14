# Package Includes
from __future__ import division

import os
import socket
import timeit
from datetime import datetime
from tensorboardX import SummaryWriter

# PyTorch includes
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

# Custom includes
from dataloaders import davis_2016 as db
from dataloaders import custom_transforms as tr
from util import visualize as viz
import scipy.misc as sm
import imageio
import networks.vgg_osvos as vo
from layers.osvos_layers import class_balanced_cross_entropy_loss
from dataloaders.helpers import *
from mypath import Path

# MAB includes
from datetime import datetime, date, time, timedelta
import sys

def force_makedirs(directory):
    """
    Create a directory (without permission error)

    Parameters
    ----------
    directory : str
        Path to the directory to create
    """
    print(directory)
    if not os.path.exists(directory):
        previous_umask = os.umask(000) # os.umask() returns the previous setting, thus previous_umask
        os.makedirs(os.path.join(directory), 0o777)
        os.umask(previous_umask)


# Documentation Reference
# https://realpython.com/documenting-python-code/
def main(sequenceName = None, parentModelName = 'parent', parentEpoch = 240, vis_net = 0, vis_res = 0):
    """
    Main Program
    Uses the OSVOS network using the fine-tuned model instead of the parent model.

    Parameters
    ----------
    sequenceName : str
        The name of the video (sequence of images) to segment
    parentModelName : str
        The name of the fined-tuned model (default 'parent')
    parentEpoch : int
        The epoch number of the model to use (default 240)
    vis_net : int [0, 1]
        Visualize the network (default 0, i.e. no)
    vis_res : int [0, 1]
        visualize the results stored at save_dir\Results (default 0, i.e. no)
    
    """

    # Setting of parameters
    if sequenceName is None:
        if 'SEQ_NAME' not in os.environ.keys():
            seq_name = 'blackswan'
        else:
            seq_name = str(os.environ['SEQ_NAME'])
    else:
        seq_name = sequenceName

    # Load the paths
    db_root_dir = Path.db_root_dir()
    save_dir = Path.save_root_dir()

    # Create the path for save_dir if it doesn't exist
    force_makedirs(save_dir)

    # User Variables
    # ------------------------------------
    # Variables that we can modify

    nAveGrad = 5  # Average the gradient every nAveGrad iterations
    nEpochs = 2000 * nAveGrad  # Number of epochs for training
    snapshot = nEpochs  # Store a model every snapshot epochs

    # Parameters in p are used for the name of the model
    p = {
        'trainBatch': 1,  # Number of Images in each mini-batch
        }
    seed = 0

    # Select which GPU, -1 if CPU
    print('CUDA is available : ' + str(torch.cuda.is_available()) + '\n')
    gpu_id = 0
    device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

    # ------------------------------------

    # Network definition
    net = vo.OSVOS(pretrained=0)
    net.load_state_dict(torch.load(os.path.join(save_dir, parentModelName+'_epoch-'+str(parentEpoch-1)+'.pth'),
                                   map_location=lambda storage, loc: storage))

    # Logging into Tensorboard
    log_dir = os.path.join(save_dir, 'runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname()+'-'+seq_name)
    writer = SummaryWriter(log_dir=log_dir)

    # Sends the code to the GPU
    net.to(device)  # PyTorch 0.4.0 style

    # Visualize the network
    if vis_net:
        x = torch.randn(1, 3, 480, 854)
        x.requires_grad_()
        x = x.to(device)
        y = net.forward(x)
        g = viz.make_dot(y, net.state_dict())
        g.view()

    # Use the following optimizer
    ##lr = 1e-8       # learning rate
    ##wd = 0.0002     # weight decay
    # Stochastic Gradient Descent (SDG) Optimizer
    # To optimize the objective function (i.e. loss function)
    ##optimizer = optim.SGD([
    ##    {'params': [pr[1] for pr in net.stages.named_parameters() if 'weight' in pr[0]], 'weight_decay': wd},
    ##    {'params': [pr[1] for pr in net.stages.named_parameters() if 'bias' in pr[0]], 'lr': lr * 2},
    ##    {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'weight' in pr[0]], 'weight_decay': wd},
    ##    {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'bias' in pr[0]], 'lr': lr*2},
    ##    {'params': [pr[1] for pr in net.upscale.named_parameters() if 'weight' in pr[0]], 'lr': 0},
    ##    {'params': [pr[1] for pr in net.upscale_.named_parameters() if 'weight' in pr[0]], 'lr': 0},
    ##    {'params': net.fuse.weight, 'lr': lr/100, 'weight_decay': wd},
    ##    {'params': net.fuse.bias, 'lr': 2*lr/100},
    ##    ], lr=lr, momentum=0.9)

    # Preparation of the data loaders
    # Define augmentation transformations as a composition
    composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                              tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
                                              tr.ToTensor()])
    
    # Training dataset and its iterator
    db_train = db.DAVIS2016(train=True, db_root_dir=db_root_dir, transform=composed_transforms, seq_name=seq_name)
    trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=1)

    # Testing dataset and its iterator
    db_test = db.DAVIS2016(train=False, db_root_dir=db_root_dir, transform=tr.ToTensor(), seq_name=seq_name)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    num_img_tr = len(trainloader)
    num_img_ts = len(testloader)
    loss_tr = []
    aveGrad = 0

    # SKIP TRAINING PART (Fine-Tuning) 
    # Since we are using the fine-tuned model directly instead of starting from the parent model to fine-tune
    #############################
    # SKIP TRAINING PART

    # Testing Phase
    # i.e. Predicting the rest of the video (sequence of images) based on the fine-tuning done during Online Training
    if vis_res:
        import matplotlib.pyplot as plt
        plt.close("all")
        plt.ion()
        f, ax_arr = plt.subplots(1, 3)

    # Create results folder for output images
    save_dir_res = os.path.join(save_dir, 'results', seq_name)
    force_makedirs(save_dir_res)

    print('Testing Network')
    # (torch.no_grad() => Do not update the gradients => Predict)
    with torch.no_grad(): 
        
        # Main Testing Loop
        for ii, sample_batched in enumerate(testloader):

            img, gt, fname = sample_batched['image'], sample_batched['gt'], sample_batched['fname']

            # Forward of the mini-batch
            inputs, gts = img.to(device), gt.to(device)

            outputs = net.forward(inputs)

            for jj in range(int(inputs.size()[0])):
                pred = np.transpose(outputs[-1].cpu().data.numpy()[jj, :, :, :], (1, 2, 0))
                pred = 1 / (1 + np.exp(-pred))
                pred = np.squeeze(pred)

                # Save the result, attention to the index jj
                # sm.imsave(os.path.join(save_dir_res, os.path.basename(fname[jj]) + '.png'), pred) # Deprecated (scipy.imsave)
                # pred_as_uint8 = pred.astype(np.uint8) # Removes Warning (DO NOT USE: OUTPUT IMAGES WON'T WORK / NO MASK)
                imageio.imwrite(os.path.join(save_dir_res, os.path.basename(fname[jj]) + '.png'), pred)
                print('Result (' + fname[jj] + '.png) Saved!')

                if vis_res:
                    img_ = np.transpose(img.numpy()[jj, :, :, :], (1, 2, 0))
                    gt_ = np.transpose(gt.numpy()[jj, :, :, :], (1, 2, 0))
                    gt_ = np.squeeze(gt)
                    # Plot the particular example
                    ax_arr[0].cla()
                    ax_arr[1].cla()
                    ax_arr[2].cla()
                    ax_arr[0].set_title('Input Image')
                    ax_arr[1].set_title('Ground Truth')
                    ax_arr[2].set_title('Detection')
                    ax_arr[0].imshow(im_normalize(img_))
                    ax_arr[1].imshow(gt_)
                    ax_arr[2].imshow(im_normalize(pred))
                    plt.pause(0.001)

    writer.close()
    print('OSVOS End..')

# Required for execution on Windows
# When you call the file (.py), it executes the main()
if __name__=='__main__':
    # Usage: perform_segmentation.py <sequenceName> <parentModelName> <parentEpoch>
    # Ex: perform_segmentation.py blackswan parent 240

    # Get command-line arguments
    sequenceName = sys.argv[1]
    parentModelName = sys.argv[2]
    parentEpoch = int(sys.argv[3])

    # Execute segmentation using model
    main(sequenceName, parentModelName, parentEpoch)