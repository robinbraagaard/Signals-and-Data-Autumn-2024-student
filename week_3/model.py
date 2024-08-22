import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from tqdm import tqdm

class FFN(torch.nn.Module):
    def __init__(self, train_loader, test_loader, in_features, num_classes, lr=0.001):
        super().__init__()
        
        # self.train_loader = train_loader
        # self.test_loader = test_loader
        self.num_classes = num_classes


        self.layers = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()

        self.optim = torch.optim.Adam(self.layers.parameters(), lr=lr)
            

    def forward(self, x):
        return self.layers(x.flatten(start_dim=1))

    def train(self, train_dataloader, epochs=1, test_dataloader=None):
        

        for epoch in tqdm(range(epochs)):

            for inputs, targets in train_dataloader:
                
                logits = self.forward(inputs)

                preds = torch.argmax(logits, dim=1)

                loss = self.criterion(logits, targets)

                loss.backward()

                self.optim.step()

                self.optim.zero_grad()
        
                

    def test(self, test_dataloader):
        
        total_acc = 0

        for input_batch, label_batch in test_dataloader:
            # Get predictions
            outs = self(input_batch)

            # Remember, outs are probabilities (so there's 10 for each input)
            # The classification the network wants to assign, must therefore be the probability with the larget value
            # We find that using argmax (dim=1, because dim=0 would be across batch dimension)
            classifications = torch.argmax(outs, dim=1)
            print(classifications)
            total_acc += (classifications == label_batch).sum().item()

        total_acc = total_acc / len(test_dataloader.dataset)

        return total_acc

    

class DiscountVGG:
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def train():
        pass


class LightningVGG:
    pass

    

dataset = 'mnist'

# TODO: PERHAPS USE THIS, PERHAPS NOT
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if dataset == 'cifar10':
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=ToTensor())

elif dataset == 'mnist':
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())



# TODO: Print some stats of the dataset here...

# DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
#            batch_sampler=None, num_workers=0, collate_fn=None,
#            pin_memory=False, drop_last=False, timeout=0,
#            worker_init_fn=None, *, prefetch_factor=2,
#            persistent_workers=False)

train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, collate_fn=None)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False, collate_fn=None)

example_input, example_label = next(iter(train_dataloader))

print("Batch size is:", len(example_input))
print("Input dim is (remember, there's also a color channel!):", example_input.shape[1:])

# TODO: Change both below to be not cancerous!
in_features = example_input.flatten(start_dim=1).shape[-1]
num_classes = 10 # 

model = FFN(None, None, in_features=in_features, num_classes=num_classes)

model.train(train_dataloader, epochs=1)


print(model.test(test_dataloader))