import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2 
from torch.utils.data import DataLoader

from parameters import parameter_reading

def load_data(args):
    bs = args.bs
    size = args.imsize
    transform_train_list = [
        v2.RandomResizedCrop(size=(size,size)),
        # v2.RandomRotation(degrees=(-10,10)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
        ]
    transform_test_list = [
        v2.Resize(size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
        ]
    
    if args.paradigm == 'ann':
        transform_train_list.append(v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        transform_test_list.append(v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    
    transform_train = v2.Compose(transform_train_list)
    transform_test = v2.Compose(transform_test_list)

    # Add RandAugment with N, M(hyperparameter)
    # if aug:
    #     N = 2
    #     M = 14
    #     transform_train.transforms.insert(0, RandAugment(N, M))

    # Prepare dataset
    trainset = CIFAR10(root='./dataset', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset,
        batch_size=bs,
        shuffle=True,
        pin_memory=True
        )

    testset = CIFAR10(root='./dataset', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, 
        batch_size=bs, 
        shuffle=False,
        pin_memory=True
        )
    return trainloader, testloader