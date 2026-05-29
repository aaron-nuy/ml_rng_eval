import torch
torch._dynamo.config.use_numpy_random_stream = False
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class BaseMNISTTrainer:
    def __init__(self, args):
        self.args = args

        self.device = self.setup_device()
        self.model = self.build_model().to(self.device)
        self.train_loader, self.test_loader = self.build_dataloaders()
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()

    def setup_device(self):
        use_accel = not self.args.no_accel and torch.accelerator.is_available()
        if use_accel:
            return torch.accelerator.current_accelerator()
        return torch.device("cpu")

    def build_model(self):
        return MNISTNet()

    def get_transforms(self, is_train=True):
        if is_train:
            return transforms.Compose([
                transforms.RandomRotation(degrees=15),
                transforms.RandomCrop(28, padding=2),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

    def get_dataloader_kwargs(self, is_train=True):
        kwargs = {'batch_size': self.args.batch_size if is_train else self.args.test_batch_size}
        use_accel = not self.args.no_accel and torch.accelerator.is_available()
        if use_accel:
            kwargs.update({
                'num_workers': 1,
                'persistent_workers': True,
                'pin_memory': True,
                'shuffle': is_train
            })
        return kwargs

    def build_dataloaders(self):
        train_transform = self.get_transforms(is_train=True)
        test_transform = self.get_transforms(is_train=False)

        full_dataset = datasets.MNIST('../data', train=True, download=True, transform=train_transform)

        val_size = 10000
        train_size = len(full_dataset) - val_size

        dataset1, dataset2 = torch.utils.data.random_split(full_dataset,[train_size, val_size])

        dataset2.dataset.transform = test_transform

        train_kwargs = self.get_dataloader_kwargs(is_train=True)
        test_kwargs = self.get_dataloader_kwargs(is_train=False)

        train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

        return train_loader, test_loader

    def build_optimizer(self):
        return optim.Adadelta(self.model.parameters(), lr=self.args.lr)

    def build_scheduler(self):
        return StepLR(self.optimizer, step_size=1, gamma=self.args.gamma)

    def train_epoch(self, epoch):
        self.model.train()
        losses = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.item()))

        return losses

    def test_epoch(self) -> tuple[int, float]:
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = float(correct) / len(self.test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * accuracy))

        return test_loss, accuracy

    def run(self):
        losses = []
        test_losses = []
        accuracies = []
        for epoch in range(1, self.args.epochs + 1):
            losses += self.train_epoch(epoch)
            test_loss, accuracy = self.test_epoch()
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            self.scheduler.step()

        return losses, test_losses, accuracies
