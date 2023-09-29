import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
from copy import deepcopy as copy
from tqdm.auto import tqdm

import os
from torchvision.models.vision_transformer import VisionTransformer


class SupervisedModel(torch.nn.Module):
    """
    supervised learning model with a .fit method similar to what is available in keras
    """
    def __init__(self):
        super(SupervisedModel, self).__init__()
        self.device = None
        self._wrapped_model = None
        
    # Training function.
    def train_step(self, trainloader, optimizer, criterion, metrics):
        self.train()
        if self.device == 0:
            print('Training')
        train_running_loss = 0.0
        train_running_correct = 0
        counter = 0
        epoch_metrics = dict()
        
        # mixed precision
        scaler = torch.cuda.amp.GradScaler()
        
        for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            counter += 1
            image, labels = data
            image = image.to(f'cuda:{self.device}', non_blocking=True)
            labels = labels.to(f'cuda:{self.device}', non_blocking=True)
            running_loss = 0
            # Forward pass.
            with torch.autocast(device_type='cuda', dtype=torch.float16):# mixed precision
                outputs = self(image)
                # Calculate the loss.
                loss = criterion(outputs, labels)
            running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            # Backpropagation
            optimizer.zero_grad()
            scaler.scale(loss).backward() # mixed precision
            # Update the weights.
            torch.distributed.barrier()
            scaler.step(optimizer) # mixed precision
            scaler.update() # mixed precision
        # Loss and accuracy for the complete epoch.
        epoch_metrics_loss = torch.Tensor([running_loss / counter]).cuda(self.device, non_blocking=True)
        dist.all_reduce(epoch_metrics_loss, dist.ReduceOp.AVG, async_op=False)
        epoch_metrics['loss'] = epoch_metrics_loss[0]
        
        for metric in metrics:
            value = torch.Tensor([metric(outputs, labels)]).cuda(self.device, non_blocking=True)
            dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
            epoch_metrics[metric.name] = value[0]
        
        return epoch_metrics

    # Validation function.
    def validate_step(self, testloader, criterion, metrics):
        self.eval()
        if self.device == 0:
            print(f'Validation')
        valid_running_loss = 0.0
        valid_running_correct = 0
        counter = 0
        epoch_metrics = dict()
        running_loss = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(testloader), total=len(testloader)):
                counter += 1

                image, labels = data

                image = image.cuda(self.device, non_blocking=True)
                labels = labels.cuda(self.device, non_blocking=True)
                # image = image.to(self.device)
                # labels = labels.to(self.device)
                # Forward pass.
                outputs = self(image)
                # Calculate the loss.
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                # Calculate the accuracy.
                _, preds = torch.max(outputs.data, 1)

        # Loss and accuracy for the complete epoch.
        # TODO: change these to more general metrics
        epoch_metrics_loss = torch.Tensor([running_loss / counter]).cuda(self.device, non_blocking=True)
        dist.all_reduce(epoch_metrics_loss, dist.ReduceOp.AVG, async_op=False)
        epoch_metrics['loss'] = epoch_metrics_loss[0]
        
        for metric in metrics:
            value = torch.Tensor([metric(outputs, labels)]).cuda(self.device, non_blocking=True)
            dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
            epoch_metrics[metric.name] = value[0]
            
        return epoch_metrics
        
    def fit(self, optimizer, criterion, epochs, train_loader, val_loader, metrics=list(), watch='val_loss', mode=min):
        """
        keras model fit for torch
        """
        if self.device == 0 or self.device is None:
            wandb.init(project='torch_test', entity='ai2es',
            config={
                'experiment': 'torch_test',
            })
        
        
        init_metrics = self.validate_step(val_loader, criterion, metrics)
        watch_metric = {'val_'+metric: init_metrics[metric] for metric in init_metrics}[watch]
        best_state = None
        
        history = []

        for epoch in range(epochs):
            train_loader.sampler.set_epoch(epoch)
            if self.device == 0 or self.device is None:
                print(f"[INFO]: Epoch {epoch+1} of {epochs}")
            train_metrics = self.train_step(train_loader, optimizer, criterion, metrics)
            val_metrics = self.validate_step(val_loader, criterion, metrics)
            
            epoch_metrics = dict()
            
            for metric in train_metrics:
                epoch_metrics['train_' + metric] = train_metrics[metric]
                
            for metric in val_metrics:
                epoch_metrics['val_' + metric] = val_metrics[metric]
            
            history.append(epoch_metrics)
            
            print_line = '\n'.join(sorted([f"{metric}: {epoch_metrics[metric]:.3f}" for metric in epoch_metrics]))
            if self.device == 0:
                print(print_line)
                print('-'*50)

            if mode(epoch_metrics[watch], watch_metric) != watch_metric:
                watch_metric = epoch_metrics[watch]
                best_state = copy(self.state_dict())
        # TODO: define the optimal metric as a model variable
        self.load_state_dict(best_state)
    

    def wrap_model_using_fsdp(self):
        params_no_grad = [n for n, p in self.named_parameters() if not p.requires_grad]
        
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            
        if len(params_no_grad) > 0:
            print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'. format(len(params_no_grad), params_no_grad))
            print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
            print("[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)
                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

        from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, MixedPrecision, ShardingStrategy
                
        dtype = torch.float16
        mixed_precision_policy = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
        device_id = int(os.environ['RANK']) % torch.cuda.device_count()
                
        def get_module_class_from_name(module, name):
            modules_children = list(module.children())
            if module.__class__.__name__ == name:
                return module.__class__
            elif len(modules_children) == 0:
                return
            else:
                for child_module in modules_children:
                    module_class = get_module_class_from_name(child_module, name)
                    if module_class is not None:
                        return module_class
                
        #from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        #import functools
        #transformer_cls_to_wrap = set()
        #auto_wrap_policy = functools.partial(
        #    transformer_auto_wrap_policy,
        #    # Transformer layer class to wrap
        #    transformer_layer_cls=VisionTransformer, # HACK: hardcoded
        #)
                
        return FSDP(
            self,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            mixed_precision=mixed_precision_policy,
            #  auto_wrap_policy=auto_wrap_policy,
            device_id=f'cuda:{device_id}'
        )

class ImageClassificationModel(SupervisedModel):
    def __init__(self, n_classes, filters=16, degree=4, pools=3, dropout=.1, image_size=224):
        super(ImageClassificationModel, self).__init__()

        from torchvision.models.vision_transformer import VisionTransformer
        self.network = VisionTransformer(
                                    image_size=image_size,
                                    patch_size=16,
                                    num_layers=12,
                                    num_heads=12,
                                    hidden_dim=768,
                                    mlp_dim=3072,
                                    num_classes=n_classes,
                                )

        self.param_layers = torch.nn.ModuleList([self.network])
    
    def forward(self, data):
        return self.network(data)
    

class SimCLR():
    def __init__(self, model, g, f, transforms) -> None:
        self.model = model
        self.g = g
        self.f = f
        self.transforms = transforms
    
    def train_step(self, trainloader, optimizer, criterion, metrics):
        self.train()
        if self.device == 0:
            print('Training')
        train_running_loss = 0.0
        train_running_correct = 0
        counter = 0
        epoch_metrics = dict()
        
        # mixed precision
        scaler = torch.cuda.amp.GradScaler()
        
        for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            counter += 1
            image, labels = data
            image = image.to(f'cuda:{self.device}', non_blocking=True)
            labels = labels.to(f'cuda:{self.device}', non_blocking=True)
            running_loss = 0
            # Forward pass.
            with torch.autocast(device_type='cuda', dtype=torch.float16):# mixed precision
                outputs = self(image)
                # Calculate the loss.
                loss = criterion(outputs, labels)
            running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            # Backpropagation
            optimizer.zero_grad()
            scaler.scale(loss).backward() # mixed precision
            # Update the weights.
            torch.distributed.barrier()
            scaler.step(optimizer) # mixed precision
            scaler.update() # mixed precision
        # Loss and accuracy for the complete epoch.
        epoch_metrics_loss = torch.Tensor([running_loss / counter]).cuda(self.device, non_blocking=True)
        dist.all_reduce(epoch_metrics_loss, dist.ReduceOp.AVG, async_op=False)
        epoch_metrics['loss'] = epoch_metrics_loss[0]
        
        for metric in metrics:
            value = torch.Tensor([metric(outputs, labels)]).cuda(self.device, non_blocking=True)
            dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
            epoch_metrics[metric.name] = value[0]
        
        return epoch_metrics

    # Validation function.
    def validate_step(self, testloader, criterion, metrics):
        self.eval()
        if self.device == 0:
            print(f'Validation')
        valid_running_loss = 0.0
        valid_running_correct = 0
        counter = 0
        epoch_metrics = dict()
        running_loss = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(testloader), total=len(testloader)):
                counter += 1

                image, labels = data

                image = image.cuda(self.device, non_blocking=True)
                labels = labels.cuda(self.device, non_blocking=True)
                # image = image.to(self.device)
                # labels = labels.to(self.device)
                # Forward pass.
                outputs = self(image)
                # Calculate the loss.
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                # Calculate the accuracy.
                _, preds = torch.max(outputs.data, 1)

        # Loss and accuracy for the complete epoch.
        # TODO: change these to more general metrics
        epoch_metrics_loss = torch.Tensor([running_loss / counter]).cuda(self.device, non_blocking=True)
        dist.all_reduce(epoch_metrics_loss, dist.ReduceOp.AVG, async_op=False)
        epoch_metrics['loss'] = epoch_metrics_loss[0]
        
        for metric in metrics:
            value = torch.Tensor([metric(outputs, labels)]).cuda(self.device, non_blocking=True)
            dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
            epoch_metrics[metric.name] = value[0]
            
        return epoch_metrics
        
    def fit(self, optimizer, criterion, epochs, train_loader, val_loader, metrics=list(), watch='val_loss', mode=min):
        """
        keras model fit for torch
        """
        
        wandb.init(project='torch_test', entity='ai2es',
           config={
               'experiment': 'torch_test',
           })
        
        
        init_metrics = self.validate_step(val_loader, criterion, metrics)
        watch_metric = {'val_'+metric: init_metrics[metric] for metric in init_metrics}[watch]
        best_state = None
        
        history = []

        for epoch in range(epochs):
            train_loader.sampler.set_epoch(epoch)
            print(f"[INFO]: Epoch {epoch+1} of {epochs}")
            train_metrics = self.train_step(train_loader, optimizer, criterion, metrics)
            val_metrics = self.validate_step(val_loader, criterion, metrics)
            
            epoch_metrics = dict()
            
            for metric in train_metrics:
                epoch_metrics['train_' + metric] = train_metrics[metric]
                
            for metric in val_metrics:
                epoch_metrics['val_' + metric] = val_metrics[metric]
            
            history.append(epoch_metrics)
            
            print_line = '\n'.join(sorted([f"{metric}: {epoch_metrics[metric]:.3f}" for metric in epoch_metrics]))
            if self.device == 0:
                print(print_line)
                print('-'*50)

            if mode(epoch_metrics[watch], watch_metric) != watch_metric:
                watch_metric = epoch_metrics[watch]
                best_state = copy(self.state_dict())
        # TODO: define the optimal metric as a model variable
        self.load_state_dict(best_state)
        
