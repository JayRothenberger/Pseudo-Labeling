from pathlib import Path
import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import wandb
import time
import pickle
import random
from copy import deepcopy as copy
import numpy as np
import gc

from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision.transforms import v2
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from distances import pairwise_cosine, pairwise_euclidean_dist, pairwise_JS, pairwise_l1_dist, pairwise_KL
from distances import query
import flash
from DAHS.DAHB import DistributedAsynchronousHyperBand, DistributedAsynchronousGridSearch
from DAHS.torch_utils import sync_parameters
from dataset_utils import find_closest_batch, add_to_imagefolder

DIST_MAP = {
    'cosine': pairwise_cosine,
    'l2': pairwise_euclidean_dist,
    'l1': pairwise_l1_dist,
    'JS': pairwise_JS,
    'KL': pairwise_KL
}

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None, weight=1.0):
    model.train()
    ddp_loss = torch.zeros(2).to(rank % torch.cuda.device_count())
    loss_fn = torch.nn.CrossEntropyLoss()
    i = 0
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(rank % torch.cuda.device_count()), target.to(rank % torch.cuda.device_count())
        optimizer.zero_grad()

        output = model(data)
        loss = loss_fn(output, target) * weight
        torch.distributed.all_reduce(loss)
        loss.backward()
        model.clip_grad_norm_(1.0, norm_type=2.0)
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)
        i+=1
        if int(os.environ["RANK"]) == 0:
            print(f'{i}/{len(train_loader)}        ', end='\r')
    i=0
    print()
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))


def train_pl(args, model, rank, world_size, train_loader, pseudo_loader, optimizer, epoch, sampler=None, weight=1.0):

    model.train()
    ddp_loss = torch.zeros(2).to(rank % torch.cuda.device_count())
    loss_fn = torch.nn.CrossEntropyLoss()
    i = 0
    if sampler:
        sampler.set_epoch(epoch)
    
    print(len(pseudo_loader), len(train_loader))

    for (batch_l, (xl, yl)), (batch_pl, (xpl, ypl)) in zip(enumerate(train_loader), enumerate(pseudo_loader)):

        xl = xl.to(rank % torch.cuda.device_count())
        xpl = xpl.to(rank % torch.cuda.device_count())
        yl = yl.to(rank % torch.cuda.device_count())
        ypl = ypl.to(rank % torch.cuda.device_count())


        optimizer.zero_grad()

        output_l = model(xl)
        output_pl = model(xpl)

        loss_l = loss_fn(output_l, yl)
        loss_pl = loss_fn(output_pl, ypl)
        loss = (loss_l + loss_pl*weight) / 2
        torch.distributed.all_reduce(loss)
        loss.backward()
        model.clip_grad_norm_(1.0, norm_type=2.0)
        optimizer.step()

        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(xl) + len(xpl)
        i += 1
        if int(os.environ["RANK"]) == 0:
            print(f'{i}/{min(len(train_loader), len(pseudo_loader))}        ')
    i=0
    print()
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))


def test(model, rank, world_size, test_loader):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    correct = 0
    ddp_loss = torch.zeros(3).to(rank % torch.cuda.device_count())
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(rank % torch.cuda.device_count()), target.to(rank % torch.cuda.device_count())
            output = model(data) / args.temperature
            ddp_loss[0] += loss_fn(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if int(os.environ["RANK"]) == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) :)\n'.format(
            test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
            100. * ddp_loss[1] / ddp_loss[2]))
    
    return ddp_loss[1] / ddp_loss[2]


def training_process(rank, world_size, args, states):
    random.seed(42)
    print(os.environ["SLURM_JOB_ID"])
    if args.from_embed is not None:
        with open(f'/ourdisk/hpc/ai2es/jroth/AI2ES_DL_Torch/statistical distances/{args.from_embed}_IN1K_train.ds', 'rb') as fp:
            train_embeds, train_labels = pickle.load(fp)

        with open(f'/ourdisk/hpc/ai2es/jroth/AI2ES_DL_Torch/statistical distances/{args.from_embed}_IN1K_val.ds', 'rb') as fp:
            val_embeds, val_labels = pickle.load(fp)

        val_embeds = torch.tensor(val_embeds)
        val_labels = torch.tensor(val_labels)

        train_embeds = torch.tensor(train_embeds)
        train_labels = torch.tensor(train_labels)

    image_size = 224
    train_transform = v2.Compose(
                            [
                                v2.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), antialias=True),
                                v2.RandomHorizontalFlip(p=0.5),
                                # transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
                                #torchvision.transforms.RandAugment(),
                                #transforms.Resize(size=(image_size, image_size), antialias=True),
                                torchvision.transforms.ToTensor(),
                                transforms.Normalize(
                                                    mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]
                                                    ),
                            ]
                            )

    test_transform = v2.Compose(
                        [
                            transforms.Resize(size=(image_size, image_size), antialias=True),
                            torchvision.transforms.ToTensor(),
                            transforms.Normalize(
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]
                                                ),
                        ]
                        )


    from torch.utils.data import default_collate
    # TODO: advanced data augmentation as is done in one of the PL papers
    #def collate_fn(batch):
    #    return torchvision.transforms.v2.CutMix(torchvision.transforms.v2.MixUp(*default_collate(batch)))

    # /scratch/jroth
    dataset1 = torchvision.datasets.ImageNet('/scratch/jroth/2012/fix', split='train', transform=train_transform)
    unlabeled_ds = torchvision.datasets.ImageNet('/scratch/jroth/2012/fix', split='train', transform=test_transform)
    pseudolabeled_ds = copy(dataset1)
    labeled_ds_train = copy(dataset1)
    
    dataset2 = torchvision.datasets.ImageNet('/scratch/jroth/2012/fix', split='val', transform=test_transform)
    
    # 10% eval, TODO: add a command line argument for this
    # TODO: class balance this selection
    
    # remember which examples we picked by their indicies in the original dataset
    if states.get('labeled_inds') is None:
        inds = np.random.choice(len(dataset1), int(len(dataset1)*args.labeled_fraction))
    else:
        inds = states.get('labeled_inds')
    
    labeled_ds_train.samples = [(str(a[0]), int(a[1])) for a in list(np.array(dataset1.samples)[inds])]

     # and the ones we did not
    if states.get('unlabeled_inds') is None:
        not_inds = np.array(list(set(np.arange(len(dataset1))) - set(inds)))
    else:
        not_inds = states.get('unlabeled_inds')

    unlabeled_samples = np.array(dataset1.samples)
    unlabeled_ds.samples = list(set(unlabeled_ds.samples) - set(labeled_ds_train.samples))

    # reconstruct the pseudo labels from the checkpoint
    if states.get('pseudolabeled_samples') is None:
        pseudolabeled_ds.samples = list()
    else:
        if args.pseudo_first:
            # add a new dataset and loader here for just pseudolabels
            pseudolabeled_ds.samples = states.get('pseudolabeled_samples')
        else:
            paths = [i for i, _ in states.get('pseudolabeled_samples')]
            labels = [j for _, j in states.get('pseudolabeled_samples')]
            labeled_ds_train = add_to_imagefolder(paths, labels, labeled_ds_train)


    # TODO: bring back in whole dataset
    # unlabeled_ds.samples = list(set(dataset1.samples) - set(labeled_ds_train.samples))
    # This will NOT be modified in the future, includes only the 'supervised' examples for the experiment
    dataset1.samples = labeled_ds_train.samples

    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=10_000_000
    )

    torch.cuda.set_device(rank % torch.cuda.device_count())

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)


    os.environ['TORCH_HOME'] = '/ourdisk/hpc/ai2es/jroth/models/'
    os.environ['TORCH_HUB'] = '/ourdisk/hpc/ai2es/jroth/models/'

    
    init_start_event.record()

    for it in range(1, args.iters + 1):
        if rank == 0:
            wandb.init(project='ISAIM', entity='ai2es',
            name=f"{rank}: Semi-Supervised",
            config={
                'experiment': 'torch_test',
                'args': vars(args),
                'it': it,
            })

        if args.pretrained:
            backbone = torch.hub.load('facebookresearch/swav:main', 'resnet50')
            if it == 1:
                backbone.requires_grad_(False)
            model = torch.nn.Sequential(
                backbone,
                torch.nn.Flatten(),
                torch.nn.Linear(1000, 1000)
            )
        else:
            model = torchvision.models.resnet50()

        print(f'{os.environ["RANK"]}: to device')

        # have to send the module to the correct device first
        model.to(rank % torch.cuda.device_count())
        print(f'{os.environ["RANK"]}: FSDP')
        model = FSDP(model, 
                    auto_wrap_policy=auto_wrap_policy,
                    mixed_precision=torch.distributed.fsdp.MixedPrecision(
                        param_dtype=torch.float16, 
                        reduce_dtype=torch.float32, 
                        buffer_dtype=torch.float32, 
                        cast_forward_inputs=True)
                    )
        # optimizer initialization must happen after wrap
        optimizer = flash.core.optimizers.LARS(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
        # learning rate schedule
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=20)
        
        sampler1 = DistributedSampler(labeled_ds_train, rank=rank, num_replicas=world_size, shuffle=True)
        sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)
        pseudo_sampler = DistributedSampler(pseudolabeled_ds, rank=rank, num_replicas=world_size, shuffle=True)

        train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
        ssl_kwargs = {'batch_size': args.batch_size}
        
        test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
        
        cuda_kwargs = {'num_workers': 12,
                        'pin_memory': True,
                        'shuffle': False}

        test_cuda_kwargs = {'num_workers': 12,
                            'pin_memory': True,
                            'shuffle': False}

        ssl_cuda_kwargs = {'num_workers': 12,
                            'pin_memory': True,
                            'shuffle': False,
                            'drop_last':True}
        
        
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(test_cuda_kwargs)
        ssl_kwargs.update(ssl_cuda_kwargs)
        pseudo_kwargs = copy(ssl_kwargs)
        pseudo_kwargs['sampler'] = pseudo_sampler

        train_loader = torch.utils.data.DataLoader(labeled_ds_train, **train_kwargs) #, collate_fn=collate_fn when we get cutmix and mixup working
        pseudo_loader = torch.utils.data.DataLoader(pseudolabeled_ds, **pseudo_kwargs) #, collate_fn=collate_fn when we get cutmix and mixup working
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
        labeled_loader = torch.utils.data.DataLoader(dataset1, **ssl_kwargs)
        unlabeled_loader = torch.utils.data.DataLoader(unlabeled_ds, **ssl_kwargs)

        states = {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
        watching = 0.0
        epochs_since_improvement = 0
        data = dict()
        # train_loader = torch.utils.data.DataLoader(labeled_ds_train, **train_kwargs)
        print(f'{os.environ["RANK"]}: Self-Training {it}')
        gc.collect()
        for epoch in range(1, args.epochs + 1):
            
            pseudo_sampler.set_epoch(epoch - 1)
            sampler1.set_epoch(epoch - 1)
            print(f'{os.environ["RANK"]}: Training')
            start_train = time.time()
            if args.pseudo_first and len(pseudolabeled_ds):
                print(f'{os.environ["RANK"]}: Pseudo-Label Pre-Training')
                train(args, model, rank, world_size, pseudo_loader, optimizer, epoch, weight=args.pseudo_weight)
                train(args, model, rank, world_size, train_loader, optimizer, epoch)
            else:
                if it > 1:
                    train_pl(args, model, rank, world_size, train_loader, pseudo_loader, optimizer, epoch, weight=args.pseudo_weight)
                else:
                    train(args, model, rank, world_size, train_loader, optimizer, epoch)
            end_train = time.time()
            start_test = time.time()
            watch = test(model, rank, world_size, test_loader)

            epochs_since_improvement += 1
            if watch > watching + 0.01:
                watching = watch
                epochs_since_improvement = 0
                states['model_state'] = model.state_dict()
                states['optimizer_state'] = optimizer.state_dict()
            data['metric'] = watch
            data['best'] = watching
            if rank == 0:
                wandb.log(data, commit=True, step=epoch)
            if epochs_since_improvement > args.patience:
                break

            scheduler.step(watch)
            end_test = time.time()
            print(f'Epoch Timing - Train: {end_train - start_train}s, Test: {end_test - start_test}s')
        start = time.time()
        # find the closest batch of examples to bring in
        if os.environ["RANK"] == '0':
            if args.from_embed is not None:
                distance_metric = DIST_MAP[args.embed_dist]
                labeled_set = train_embeds[inds]
                unlabeled_set = train_embeds[not_inds]
                print(unlabeled_set.shape)
                # want to find the args.pseudolabels examples in unlabeled set which minimize the sum distance to the k closest neighbors 
                # block up this computation to mitigate U*L space complexity
                partition_distances = []
                parts = 32
                for part in range(0, parts):
                    gc.collect()
                    partition = slice(part*(unlabeled_set.shape[0] // parts), (part + 1)*(unlabeled_set.shape[0] // parts))
                    if part*(unlabeled_set.shape[0] // parts) >= unlabeled_set.shape[0] - 1:
                        break
                    rankings, distances = query(unlabeled_set[partition], labeled_set, return_size=args.k, retrieval_metric=DIST_MAP[args.embed_dist], descending=False)
                    # each row of distances is a query, each column a value
                    partition_distances.append(distances.sum(-1))

                pseudo_inds = np.array(list(set(torch.concat(partition_distances, 0).argsort()[:args.pseudolabels]))).astype(np.int32)
                print(pseudo_inds.shape, torch.concat(partition_distances, 0).argsort().shape)
                closest_batch = [(pseudo_inds[i], int(a[1])) for i, a in enumerate(list(unlabeled_samples[pseudo_inds]))]
                
                not_inds = sorted(list(set(not_inds) - set(pseudo_inds)))

                states['unlabeled_inds'] = not_inds
                states['labeled_inds'] = inds

                del unlabeled_set
                del labeled_set
                del rankings
                del distances
                gc.collect()
            else:
                closest_batch = find_closest_batch(labeled_loader, unlabeled_loader, lambda x, y: torch.sum(torch.flatten(x - y, 2), -1), m=args.pseudolabels, nearest_neighbors=1)
            samples_record = {'closest_batch': closest_batch}
            # write the new file list to a file
            with open(f'/ourdisk/hpc/ai2es/jroth/global_samples_list_{os.environ["SLURM_JOB_ID"]}', 'wb') as fp:
                pickle.dump(samples_record, fp)

        # wait until the file has been written to (dist.barrier with nccl does not work with cpu synchronization)
        while os.path.getctime(f'/ourdisk/hpc/ai2es/jroth/global_samples_list_{os.environ["SLURM_JOB_ID"]}') < start:
            time.sleep(10)
        print(f'{os.environ["RANK"]}: loading new file list')
        
        with open(f'/ourdisk/hpc/ai2es/jroth/global_samples_list_{os.environ["SLURM_JOB_ID"]}', 'rb') as fp:
            with torch.no_grad():
                samples_record = pickle.load(fp)
                closest_batch = samples_record['closest_batch']
                # update the dataset
                paths = [unlabeled_ds.samples[index][0] for value, index in closest_batch]
                # predict pseudolabels
                labels_tensors = []
                labels = []
                model.load_state_dict(states['model_state'])
                for value, index in closest_batch:
                    local_labels = torch.argmax(model(unlabeled_ds[index][0].to(rank % torch.cuda.device_count()).unsqueeze(0)), -1)
                    global_labels = [local_labels for i in range(int(os.environ["WORLD_SIZE"]))]
                    dist.all_gather(global_labels, local_labels)
                    global_labels = [int(label) for label in global_labels]
                    labels += global_labels
                # labels = [int(torch.concat(dist.all_gather(labels_tensors, torch.argmax(model(unlabeled_ds[index][0].to(rank % torch.cuda.device_count()).unsqueeze(0)), -1))).cpu()) for value, index in closest_batch]
                unlabeled_ds.samples = list(set(unlabeled_ds.samples) - set(labeled_ds_train.samples))

                if states.get('pseudolabeled_samples') is not None:
                    states['pseudolabeled_samples'] += list(zip(paths, labels))
                else:
                    states['pseudolabeled_samples'] = list(zip(paths, labels))

                pseudolabeled_ds = add_to_imagefolder(paths, labels, pseudolabeled_ds)


                print(f'{os.environ["RANK"]}: train - {len(labeled_ds_train)}')
        if rank == 0:
            wandb.finish()

    init_end_event.record()

    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")

    # use a barrier to make sure training is done on all ranks
    dist.barrier()
    # state_dict for FSDP model is only available on Nightlies for now

    return states, watching


def fsdp_main(rank, world_size, args):

    setup(rank, world_size)
    print('defining agent')

    Path(f'/ourdisk/hpc/ai2es/jroth/global_samples_list_{os.environ["SLURM_JOB_ID"]}').touch()
    
    args.pseudo_weight = [0.5, 0.25, 0.125, 2**(-4)]
    args.oracle = [True, False]
    args.from_embed = ['SwAV', 'Hist']
    args.pretrained = [True, False]

    search_space = ['pseudo_weight', 'oracle', 'from_embed', 'pretrained']

    agent = sync_parameters(args, rank, search_space, DistributedAsynchronousGridSearch)

    args = agent.to_namespace(agent.combination)

    try:
        states = torch.load(agent.path)
        print(agent.path)
    except Exception as e:
        print(agent.path, e)
        states = {'model_state': None, 'optimizer_state': None}

    states, metric = training_process(rank, world_size, args, states)

    if rank == 0:
        print('saving checkpoint')
        agent.save_checkpoint(states)

    print('finishing combination')
    agent.finish_combination(float(metric))

    print('cleanup')
    cleanup()
    exit()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Example')
    parser.add_argument('--batch_size', type=int, default=512, # NEW
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=512,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=196,
                        help='number of epochs to train per iteration (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--iters', type=int, default=5,
                        help='Self-Training Iterations - 1 means the model is retrained one time(s) (default: 1)')
    parser.add_argument('--pseudolabels', type=int, default=1024*128,
                        help='Number of pseudolabels to compute from the unlabeled set')
    parser.add_argument('--pseudo_first', action='store_true', default=False,
                        help='Train on pseudolabels before training on supervised label in each epoch')
    parser.add_argument('--pseudo_weight', type=float, default=0.1,
                        help='weight for pseudo-labeled batch when training simultaneously on labeled and pseudo-labeled examples')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--from_embed', type=str, default='DINOv2', metavar='e',
                        help='embeddings over which to compute the distances')
    parser.add_argument('--embed_dist', type=str, default='cosine', metavar='e',
                        help='distance metric over embedded vectors')
    parser.add_argument('--k', type=int, default=3, metavar='e',
                        help='nearest neighbors over which sum of distances is computed')
    parser.add_argument('--labeled_fraction', type=float, default=0.1,
                        help='percent of labeled data to use for training')
    parser.add_argument('--oracle', action='store_true', default=False,
                        help='Use the distance metric as an oracle.')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Use the pretrained backbone.')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Cross Entropy Temperature')
    parser.add_argument('--patience', type=int, default=30,
                        help='number of epochs to train for without improvement')
    parser.add_argument('--path', type=str, default='/ourdisk/hpc/ai2es/jroth/AI2ES_DL_Torch/Self-Supervised/Grid-ISAIM',
                        help='path for hparam search directory')
                        
                        
    args = parser.parse_args()

    torch.manual_seed(42)

    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    fsdp_main(int(os.environ["RANK"]), WORLD_SIZE, args)
