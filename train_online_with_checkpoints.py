# Package Includes
from __future__ import division

import os
import socket
import timeit
from datetime import datetime, date, time, timedelta
#from tensorboardX import SummaryWriter

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

# Custom includes (MAB)
from myconfigs import Configs


def main():
    """
    MAIN() PROGRAM
    Instructions :

    * To configure the paths used by this program, change the values in the file mypath.py
    * To configure the user parameters used by this program, change the values in the file myconfigs.py

    """

    # Setting of parameters
    if 'SEQ_NAME' not in os.environ.keys():
        seq_name = Configs.sequence_name        # default: blackswan
    else:
        seq_name = str(os.environ['SEQ_NAME'])

    # Load the paths
    db_root_dir = Path.db_root_dir()
    save_dir = Path.save_root_dir()

    # Create the path for save_dir if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))

    # Select which GPU, -1 if CPU
    print('CUDA is available : ' + str(torch.cuda.is_available()) + '\n')
    gpu_id = 0
    device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

    # ------------------------------------

    # Network definition
    net = vo.OSVOS(pretrained=0)
    net.load_state_dict(torch.load(os.path.join(save_dir, Configs.parentModelName + '_epoch-' + str(Configs.parentEpoch - 1) +'.pth'),
                                   map_location=lambda storage, loc: storage))

    # Logging into Tensorboard
    log_dir = os.path.join(save_dir, 'runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname()+'-'+seq_name)
    writer = SummaryWriter(log_dir=log_dir)

    # Sends the code to the GPU
    net.to(device)  # PyTorch 0.4.0 style

    # Visualize the network
    if Configs.vis_network:
        x = torch.randn(1, 3, 480, 854)
        x.requires_grad_()
        x = x.to(device)
        y = net.forward(x)
        g = viz.make_dot(y, net.state_dict())
        g.view()

    # Use the following optimizer
    lr = 1e-8       # learning rate
    wd = 0.0002     # weight decay
    # Stochastic Gradient Descent (SDG) Optimizer
    # To optimize the objective function (i.e. loss function)
    optimizer = optim.SGD([
        {'params': [pr[1] for pr in net.stages.named_parameters() if 'weight' in pr[0]], 'weight_decay': wd},
        {'params': [pr[1] for pr in net.stages.named_parameters() if 'bias' in pr[0]], 'lr': lr * 2},
        {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'weight' in pr[0]], 'weight_decay': wd},
        {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'bias' in pr[0]], 'lr': lr*2},
        {'params': [pr[1] for pr in net.upscale.named_parameters() if 'weight' in pr[0]], 'lr': 0},
        {'params': [pr[1] for pr in net.upscale_.named_parameters() if 'weight' in pr[0]], 'lr': 0},
        {'params': net.fuse.weight, 'lr': lr/100, 'weight_decay': wd},
        {'params': net.fuse.bias, 'lr': 2*lr/100},
        ], lr=lr, momentum=0.9)

    # Preparation of the data loaders
    # Define augmentation transformations as a composition
    composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                              tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
                                              tr.ToTensor()])
    
    # Training dataset and its iterator
    db_train = db.DAVIS2016(train=True, db_root_dir=db_root_dir, transform=composed_transforms, seq_name=seq_name)
    trainloader = DataLoader(db_train, batch_size=Configs.p['trainBatch'], shuffle=True, num_workers=1)

    # Testing dataset and its iterator
    db_test = db.DAVIS2016(train=False, db_root_dir=db_root_dir, transform=tr.ToTensor(), seq_name=seq_name)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    # Training Phase
    # i.e. Fine-tuning using the first frame
    train(device, seq_name, 0, Configs.nb_epochs, net, optimizer, trainloader, testloader, writer, save_dir)    

    # Testing Phase
    # i.e. Predicting the rest of the video (sequence of images) based on the fine-tuning done during Online Training
    test(device, seq_name, net, testloader, writer, save_dir)

    writer.close()
    print('OSVOS End..')

def train(device, seq_name, start_epoch, end_epoch, network, optimizer, trainloader, testloader, writer, save_dir):
    num_img_tr = len(trainloader)
    num_img_ts = len(testloader)
    loss_tr = []
    aveGrad = 0    
    
    print("\nStart of Online Training (fine-tuning Phase), sequence: " + seq_name)
    print("Started at: ", datetime.now())
    start_time = timeit.default_timer()

    # Main Training and Testing Loop
    # i.e. Fine-tuning using the first image of video (sequence of images)
    for epoch in range(start_epoch, end_epoch):

        # One training epoch
        running_loss_tr = 0
        np.random.seed(Configs.seed + epoch)
        
        for ii, sample_batched in enumerate(trainloader):

            inputs, gts = sample_batched['image'], sample_batched['gt']

            # Forward-Backward of the mini-batch
            inputs.requires_grad_() # Training -> Requires gradients to be computed
            inputs, gts = inputs.to(device), gts.to(device) # Push the images and ground-truths to GPU

            outputs = network.forward(inputs)

            # Compute the fuse loss
            loss = class_balanced_cross_entropy_loss(outputs[-1], gts, size_average=False)
            running_loss_tr += loss.item()  # PyTorch 0.4.0 style

            # Print stuff
            if epoch % (Configs.nb_epochs // 20) == (Configs.nb_epochs // 20 - 1):
                running_loss_tr /= num_img_tr
                loss_tr.append(running_loss_tr)

                print('[Epoch: %d, numImages: %5d]' % (epoch+1, ii + 1))
                print('Loss: %f' % running_loss_tr)
                writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)

            # Backward the averaged gradient
            loss /= Configs.avg_gradient_every
            loss.backward()
            aveGrad += 1

            # Update the weights once in nAveGrad forward passes
            if aveGrad % Configs.avg_gradient_every == 0:
                writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

        # Save the model
        if (epoch % Configs.checkpoint_every) == (Configs.checkpoint_every - 1) and (epoch != 0):
            torch.save(network.state_dict(), os.path.join(save_dir, seq_name + '_epoch-'+str(epoch) + '.pth'))
            # todo try checkpoint function


    stop_time = timeit.default_timer()

    print('Online Training - Time (seconds): ' + str(stop_time - start_time))
    print('Time (HH:MM:SS): ', timedelta(seconds=(stop_time - start_time)))
    print("Ended at: ", datetime.now())

def test(device, seq_name, network, testloader, writer, save_dir) :
    if Configs.vis_results:
        import matplotlib.pyplot as plt
        plt.close("all")
        plt.ion()
        f, ax_arr = plt.subplots(1, 3)

    # Defines the path where to save the results, create the directory if it doesn't exist
    save_dir_res = os.path.join(save_dir, 'Results', seq_name)
    if not os.path.exists(save_dir_res):
        os.makedirs(save_dir_res)

    print('Testing Network')
    with torch.no_grad():  # PyTorch 0.4.0 style (Do not update the gradients = Predict)
        
        # Main Testing Loop
        for ii, sample_batched in enumerate(testloader):

            img, gt, fname = sample_batched['image'], sample_batched['gt'], sample_batched['fname']

            # Forward of the mini-batch
            inputs, gts = img.to(device), gt.to(device)

            outputs = network.forward(inputs)

            for jj in range(int(inputs.size()[0])):
                pred = np.transpose(outputs[-1].cpu().data.numpy()[jj, :, :, :], (1, 2, 0))
                pred = 1 / (1 + np.exp(-pred))
                pred = np.squeeze(pred)

                # Save the result, attention to the index jj
                # sm.imsave(os.path.join(save_dir_res, os.path.basename(fname[jj]) + '.png'), pred) # Deprecated (scipy.imsave)
                pred_as_uint8 = pred.astype(np.uint8) # Removes Warning
                imageio.imwrite(os.path.join(save_dir_res, os.path.basename(fname[jj]) + '.png'), pred_as_uint8)
                print('Result (' + fname[jj] + '.png) Saved!')

                if Configs.vis_results:
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

# Required for execution on Windows
# When you call train_online.py, it executes the main()
if __name__=='__main__':
    main()