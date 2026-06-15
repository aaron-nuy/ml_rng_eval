import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

FASHION_MNIST_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

FASHION_MEAN = (0.2860,)
FASHION_STD  = (0.3530,)


class FashionMNISTNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1,   32, 3, padding=1)
        self.conv2 = nn.Conv2d(32,  32, 3, padding=1)

        self.conv3 = nn.Conv2d(32,  64, 3, padding=1)
        self.conv4 = nn.Conv2d(64,  64, 3, padding=1)

        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)

        self.dropout1 = nn.Dropout(0.40)
        self.dropout2 = nn.Dropout(0.60)

        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = F.relu(self.conv5(x))
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


class BaseFashionMNISTTrainer:
    def __init__(self, args):
        self.args = args
        self.device       = self.setup_device()
        self.model        = self.build_model().to(self.device)
        self.train_loader, self.test_loader = self.build_dataloaders()
        self.optimizer    = self.build_optimizer()
        self.scheduler    = self.build_scheduler()

    def setup_device(self):
        use_accel = not self.args.no_accel and torch.accelerator.is_available()
        if use_accel:
            return torch.accelerator.current_accelerator()
        return torch.device("cpu")

    def build_model(self):
        return FashionMNISTNet()


    def get_transforms(self, is_train=True):
        if is_train:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.RandomCrop(28, padding=2),
                transforms.ToTensor(),
                transforms.Normalize(FASHION_MEAN, FASHION_STD)
            ])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MEAN, FASHION_STD)
        ])

    def get_dataloader_kwargs(self, is_train=True):
        kwargs = {'batch_size': self.args.batch_size if is_train else self.args.test_batch_size}
        use_accel = not self.args.no_accel and torch.accelerator.is_available()
        if use_accel:
            kwargs.update({
                'num_workers': 1,
                'persistent_workers': True,
                'pin_memory': True,
                'shuffle': is_train,
            })
        return kwargs

    def build_dataloaders(self):
        train_transform = self.get_transforms(is_train=True)
        test_transform  = self.get_transforms(is_train=False)

        train_dataset = datasets.FashionMNIST('../data', train=True, download=True,
                                              transform=train_transform)
        val_dataset   = datasets.FashionMNIST('../data', train=True, download=True,
                                              transform=test_transform)

        val_size   = 10000
        train_size = len(train_dataset) - val_size
        indices      = torch.randperm(len(train_dataset)).tolist()
        train_subset = torch.utils.data.Subset(train_dataset, indices[:train_size])
        val_subset   = torch.utils.data.Subset(val_dataset,   indices[train_size:])

        train_loader = torch.utils.data.DataLoader(train_subset,
                                                   **self.get_dataloader_kwargs(is_train=True))
        test_loader  = torch.utils.data.DataLoader(val_subset,
                                                   **self.get_dataloader_kwargs(is_train=False))
        return train_loader, test_loader

    def build_optimizer(self):
        return optim.Adam(self.model.parameters(),
                          lr=self.args.lr, weight_decay=1e-4)

    def build_scheduler(self):
        return CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)

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

    def test_epoch(self) -> tuple[float, float]:
        self.model.eval()
        test_loss = 0
        correct   = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output     = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred       = output.argmax(dim=1, keepdim=True)
                correct   += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy   = float(correct) / len(self.test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset), 100. * accuracy))
        return test_loss, accuracy

    def run(self):
        losses, test_losses, accuracies = [], [], []
        for epoch in range(1, self.args.epochs + 1):
            losses += self.train_epoch(epoch)
            test_loss, accuracy = self.test_epoch()
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            self.scheduler.step()
        return losses, test_losses, accuracies

    def predict(self, image: torch.Tensor) -> tuple[int, str]:
        self.model.eval()
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        with torch.no_grad():
            output = self.model(image)
            print(F.softmax(output, dim=1))
            idx = output.argmax(dim=1).item()
        return idx, FASHION_MNIST_CLASSES[idx]

    def save_model(self, path: str):
        torch.save({
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }, path)

    def load_model(self, path: str, map_location=None):
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.model.to(self.device)