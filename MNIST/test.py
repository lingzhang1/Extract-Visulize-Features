import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU(True)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.relu_fc1 = nn.ReLU(True)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, img):

        conv1_feat = self.conv1(img)
        conv1_feat = self.pool1(conv1_feat)
        conv1_feat = self.relu1(conv1_feat)
        conv2_feat = self.conv2(conv1_feat)
        conv2_feat = self.pool2(conv2_feat)
        conv2_feat = self.relu2(conv2_feat)


        conv2_feat = conv2_feat.view(-1, 320)
        fc1_feat = self.fc1(conv2_feat)
        fc1_feat = self.relu_fc1(fc1_feat)
        fc1_feat = F.dropout(fc1_feat, training=self.training)
        fc2_feat = self.fc2(fc1_feat)
        return F.log_softmax(fc2_feat, dim=1)

def test(args, model, device, test_loader):

    #set the model into evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #send the data into model and get the output
            output = model(data)

            #loss is no need for the testing
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')


    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    ##############################################################################
    # load the saved model
    ##############################################################################
    model = torch.load('./mnist_cnn_model.pk')
    # model = Net().to(device)
    # model.load_state_dict(torch.load('./mnist_cnn_model.pk'))
    
    
    ##############################################################################
    # testing the saved model
    ##############################################################################

    test(args, model, device, test_loader)

if __name__ == '__main__':
    main()