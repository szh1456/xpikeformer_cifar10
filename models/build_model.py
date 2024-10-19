import torch
import os
from models.get_rpu import get_rpu
from aihwkit.nn.conversion import convert_to_analog

def build_model(args):
    model_name = args.net
    size = args.imsize
    patch = args.patchsize
    if args.paradigm == "ann":
        from models.ann.transformer1 import ViT
    else:
        from models.snn.transformer import SViT as ViT

    if model_name == "vit_small":
        embed_dim = 512
        net = ViT(
            paradigm = args.paradigm,
            model_name = 'vit_small',
            encoding = None if args.paradigm == "ann" else args.encoding,
            image_size=size,
            patch_size=patch,
            num_classes=10,
            dim=embed_dim,
            depth=6,
            heads=8,
            dropout=0.1,
            emb_dropout=0.1
        )
    elif model_name == "vit_tiny":
        embed_dim = 256
        net = ViT(
            paradigm = args.paradigm,
            model_name = 'vit_tiny',
            encoding = None if args.paradigm == "ann" else args.encoding,
            image_size=size,
            patch_size=patch,
            num_classes=10,
            dim=embed_dim,
            depth=4,
            heads=8,
            dropout=0.1,
            emb_dropout=0.1
        )
    elif model_name == "vit_base":
        embed_dim = 768
        net = ViT(
            paradigm = args.paradigm,
            model_name = 'vit_base',
            encoding = None if args.paradigm == "ann" else args.encoding,
            image_size=size,
            patch_size=patch,
            num_classes=10,
            dim=embed_dim,
            depth=16,
            heads=12,
            dropout=0.1,
            emb_dropout=0.1
        )
    elif model_name == "vit_large":
        embed_dim = 1024
        net = ViT(
            paradigm = args.paradigm,
            model_name = 'vit_large',
            encoding = None if args.paradigm == "ann" else args.encoding,
            image_size=size,
            patch_size=patch,
            num_classes=10,
            dim=embed_dim,
            depth=24,
            heads=16,
            dropout=0.1,
            emb_dropout=0.1
        )
    print(f'Model: {net.model_name}, implemenataion: {args.paradigm}\n\timage size: {net.image_size}\n\tpatch size: {net.patch_size}\n\tembedding dimeinson: {net.dim}\n\tnumber of layers: {net.depth}\n\tnumber of heads: {net.heads}')
    return net

def load_model(model, args, analog=False):
    folder_path = f'./model_parameters/{"analog/" if analog else ""}'
    if args.paradigm != 'ann':
        path = f"{folder_path}{args.paradigm}_{args.net}_{args.encoding}.pth"
        subpath = _swap_words(path,'_b.pth','_u.pth')
    else:
        path = f"{folder_path}{args.paradigm}_{args.net}.pth"

    if os.path.exists(path):
        state_dict = torch.load(path)

        # layers_to_skip = ['to_patch_embedding.to_patch_tokens.1.weight', 'to_patch_embedding.to_patch_tokens.1.bias']
        # for layer in layers_to_skip:
        #     if layer in state_dict:
        #         del state_dict[layer]

        model.load_state_dict(state_dict, strict=False)
        # model.load_state_dict(stat_dict, strict=False)
        print(f'Model parameters loaded: {path}')
    else:
        print("Model file does not exist at the specified path.")
        if args.paradigm != 'ann':
            if os.path.exists(subpath):
                model.load_state_dict(torch.load(subpath), strict=False)
                print(f'Model parameters substitue loaded: {subpath}') 
    return model

def _swap_words(s, A, B):
    placeholder = "{placeholder}"
    s = s.replace(A, placeholder)
    s = s.replace(B, A)
    s = s.replace(placeholder, B)   
    return s

def convert_model(model,args,rpu='pcm'):
    rpu_config = get_rpu(rpu)
    model = convert_to_analog(model,rpu_config)
    return model

