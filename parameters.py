import argparse

def parameter_reading():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Task')
    parser.add_argument('--mode', default='train', choices=['test', 'train'], help='train/test phase')
    parser.add_argument('--paradigm', default='snn', choices=['ann', 'snn', 'ssa'], help='computing paradigm')
    parser.add_argument('--encoding', default='u', choices=['u', 'b'], help='uniform (u) or Bernoulli (b) encoding')
    parser.add_argument('--net', default='vit_small', choices=['vit_small', 'vit_tiny', 'vit_base', 'vit_large'], help='computing paradigm')
    parser.add_argument('--imsize', default=32, type=int, help='image size (pixels per edge)')
    parser.add_argument('--patchsize', default=4, type=int, help='patch size (per edge)')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') 
    parser.add_argument('--bs', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', type=int, default='200')
    parser.add_argument('--n_timesteps', type=int, default=20)
    parser.add_argument('--analog', action='store_true', help='Use analog computing')
    args = parser.parse_args()
    return args
