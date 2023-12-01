import os
import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms


from transforms import *
from dataset import *
from model import UNet


class Trainer:

    def __init__(self, data_dict):

        logging.info('initializing Trainer class')
        # check if we have  a gpu
        if torch.cuda.is_available():
            print("GPU is available")
            self.device = torch.device("cuda:0")
        else:
            print("GPU is not available")
            self.device = torch.device("cpu")

        self.dir_train = data_dict['dir_train']
        self.dir_test = data_dict['dir_test']
        self.dir_results = data_dict['dir_results']
        self.dir_checkpoints = data_dict['dir_checkpoints']

        self.lr = data_dict['lr']
        self.beta1 = data_dict['beta1']
        self.beta2 = data_dict['beta2']

        self.num_epochs = data_dict['num_epochs']
        self.epoch_save_freq = data_dict['epoch_save_freq']
        self.epoch_to_load = data_dict['epoch_to_load']


    def save(self, dir_checkpoints, net, epoch):
        if not os.path.exists(dir_checkpoints):
            os.makedirs(dir_checkpoints)

        torch.save(net.state_dict(),
                    '%s/model_epoch%04d.pth' % (dir_checkpoints, epoch))


    def load(self, dir_checkpoints, network, epoch=None):
        """Loads a PyTorch model from a checkpoint file.

        Args:
            dir_checkpoints: The path to the directory containing the checkpoint files.
            epoch: The epoch number of the checkpoint file to load. If `epoch` is `None`, the latest checkpoint file will be loaded.

        Returns:
            A PyTorch model.
        """

        # Check if the checkpoint directory exists.
        if not os.path.exists(dir_checkpoints):
            raise FileNotFoundError(f"Checkpoint directory does not exist: {dir_checkpoints}")

        # Get the list of checkpoint files in the directory.
        checkpoint_files = os.listdir(dir_checkpoints)

        # If `epoch` is `None`, load the latest checkpoint file.
        if epoch is None:
            checkpoint_file = checkpoint_files[-1]
            print(f"Loading the latest checkpoint file: {checkpoint_file}")
        else:
            checkpoint_file = f"model_epoch{epoch:04d}.pth"

        # Check if the checkpoint file exists.
        checkpoint_path = os.path.join(dir_checkpoints, checkpoint_file)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

        # Load the model state dictionary from the checkpoint file.
        model_state_dict = torch.load(checkpoint_path)

        # Create a new PyTorch model.
        network.load_state_dict(model_state_dict)

        return network


    def save_image(self, data, file_path):
        """
        Save a PyTorch tensor as an image file.

        Parameters:
        - data: PyTorch tensor
        - file_path: File path to save the image
        """
        # Move the tensor to CPU if it's on CUDA
        if data.is_cuda:
            data = data.cpu()

        # Convert PyTorch tensor to NumPy array
        image_array = data.numpy()

        # Save the image using matplotlib
        plt.imsave(file_path, image_array.squeeze(), cmap='gray')


    def train(self):

        # generate Transformations

        transform_train = transforms.Compose([
            ToNumpyArray(),
            NormalizeArray(),
            RandomFlip(),
            RandomCrop(),
            GenerateN2VMask(),
            ToTensor()
        ])

        # inv_transform


        # generate Dataset
        logging.info(self.dir_train)
        dataset_train = Dataset3D(self.dir_train, transform=transform_train)

        # make loaders

        loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=8, shuffle=True)

        # setup network

        Network = UNet().to(self.device)

        # setup loss

        l1_loss = nn.L1Loss().to(self.device)

        # setup optimization

        params = Network.parameters()

        # start from checkpoint (later implementation)

        start_epoch = 0

        optim = torch.optim.Adam(params, lr=self.lr, betas=(self.beta1, self.beta2))

        #### Training ####
        logging.info('starting training loop')
        for epoch in range(start_epoch + 1, self.num_epochs + 1):

            ### training phase
            logging.info('TRAIN: EPOCH %d' % (epoch))

            Network.train()

            for batch, data in enumerate(loader_train, 1):

                input_net, label, mask = data['input'].to(self.device), data['label'].to(self.device), data['mask'].to(self.device)

                # forward net
                # print('forwarding net')
                output_net = Network(input_net)

                # backward net
                optim.zero_grad()
                loss = l1_loss(output_net * (1-mask), label * (1-mask))
                loss.backward()
                optim.step()

                # print('TRAIN: EPOCH %d: BATCH %04d: LOSS: %.4f: '
                #       % (epoch, batch, loss))
                
            if (epoch % self.epoch_save_freq) == 0:
                self.save(self.dir_checkpoints, Network, epoch)


    def test(self):

        # setup transforms

        transform_test = transforms.Compose([
            ToNumpyArray(),
            NormalizeArray(),
            GenerateN2VMask(),
            ToTensor()
        ])

        inv_transform_test = transforms.Compose([
            BackToNumpyArray()
        ])

        # setup dataset

        dataset_test = Dataset3D(self.dir_test, transform=transform_test)
        loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=8, shuffle=False)

        # load network

        Network = UNet().to(self.device)

        Network = self.load(self.dir_checkpoints, Network, epoch=self.epoch_to_load)

        # setup loss

        l1_loss = nn.L1Loss().to(self.device)


        ## test phase

        with torch.no_grad():

            Network.eval()

            for batch, data in enumerate(loader_test, 1):

                input_net, label, mask = data['input'].to(self.device), data['label'].to(self.device), data['mask'].to(self.device)
                output_net = Network(input_net)

                loss = l1_loss(output_net, label)
                input_net, label, output_net = inv_transform_test(input_net), inv_transform_test(label), inv_transform_test(output_net)

                for j in range(label.shape[0]):
                    name = label.shape[0] * (batch-1) + j

                    fileset = {'name': name,
                               'input': "%04d-input.png" % name,
                               'output': "%04d-output.png" % name,
                               'label': "%04d-label.png" % name,
                               'clean': "%04d-clean.png" % name
                               }
                    
                    plt.imsave(os.path.join(self.dir_results, fileset['input']), input_net[j, 0, :, :].squeeze(), cmap='gray')
                    plt.imsave(os.path.join(self.dir_results, fileset['output']), output_net[j, 0, :, :].squeeze(), cmap='gray')
                    plt.imsave(os.path.join(self.dir_results, fileset['label']), label[j, 0, :, :].squeeze(), cmap='gray')

                logging.info('TEST: %d: LOSS: %.6f' % (batch, loss.item()))


