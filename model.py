import torch
import torchvision
import torch.nn.functional as F
from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import tensorflow as tf
import gc
import time
import torch.nn as nn
from torchvision.transforms import Compose
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
printing_freq = 20

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class ModelWrapper(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        self.model = None
        self.model_name = None

    @abstractmethod
    def get_cutted_model(self, bottleneck):
        pass

    def get_gradient(self, acts, y, bottleneck_name):
        inputs = torch.autograd.Variable(torch.tensor(acts).to(device), requires_grad=True)
        targets = (y[0] * torch.ones(inputs.size(0))).long().to(device)

        cutted_model = self.get_cutted_model(bottleneck_name).to(device)
        cutted_model.eval()
        outputs = cutted_model(inputs)

        # y=[i]
        grads = -torch.autograd.grad(outputs[:, y[0]], inputs)[0]
        
        grads = grads.detach().cpu().numpy()

        cutted_model = None
        gc.collect()

        return grads

    def reshape_activations(self, layer_acts):
        return np.asarray(layer_acts).squeeze()

    @abstractmethod
    def label_to_id(self, label):
        pass

    def run_examples(self, examples, bottleneck_name):

        # print('Model run_examples running......')
        global bn_activation
        bn_activation = None

        def save_activation_hook(mod, inp, out):
            global bn_activation
            bn_activation = out

        handle = self.model._modules[bottleneck_name].register_forward_hook(save_activation_hook)

        self.model.to(device)
        # print('Inputs before permute: ', examples.shape)
        inputs = torch.FloatTensor(examples).permute(0, 3, 1, 2).to(device)
        # print('Inputs after permute: ', inputs.shape)
        self.model.eval()
        print('Model evaluation begins...')
        self.model(inputs)
        print('Model evaluation ends...\n')
        acts = bn_activation.detach().cpu().numpy()
        handle.remove()
        # print("Activations generated: ", acts, '\n')

        return acts


class ImageModelWrapper(ModelWrapper):
    """Wrapper base class for image models."""

    def __init__(self, image_shape):
        super(ModelWrapper, self).__init__()
        # shape of the input image in this model
        self.image_shape = image_shape

    def get_image_shape(self):
        """returns the shape of an input image."""
        return self.image_shape


class PublicImageModelWrapper(ImageModelWrapper):
    """Simple wrapper of the public image models with session object.
    """

    def __init__(self, labels_path, image_shape):
        super(PublicImageModelWrapper, self).__init__(image_shape=image_shape)
        self.labels = tf.gfile.Open(labels_path).read().splitlines()
        print(self.labels)

    def label_to_id(self, label):
        return self.labels.index(label)


class InceptionV3_cutted(torch.nn.Module):
    def __init__(self, inception_v3, bottleneck):
        super(InceptionV3_cutted, self).__init__()
        names = list(inception_v3._modules.keys())
        layers = list(inception_v3.children())

        self.layers = torch.nn.ModuleList()
        self.layers_names = []

        bottleneck_met = False
        for name, layer in zip(names, layers):
            if name == bottleneck:
                bottleneck_met = True
                continue  # because we already have the output of the bottleneck layer
            if not bottleneck_met:
                continue
            if name == 'AuxLogits':
                continue

            self.layers.append(layer)
            self.layers_names.append(name)

    def forward(self, x):
        y = x
        for i in range(len(self.layers)):
            # pre-forward process
            if self.layers_names[i] == 'Conv2d_3b_1x1':
                y = F.max_pool2d(y, kernel_size=3, stride=2)
            elif self.layers_names[i] == 'Mixed_5b':
                y = F.max_pool2d(y, kernel_size=3, stride=2)
            elif self.layers_names[i] == 'fc':
                y = F.adaptive_avg_pool2d(y, (1, 1))
                y = F.dropout(y, training=self.training)
                y = y.view(y.size(0), -1)

            y = self.layers[i](y)
        return y


class InceptionV3Wrapper(PublicImageModelWrapper):

    def __init__(self, labels_path):
        image_shape = [299, 299, 3]
        super(InceptionV3Wrapper, self).__init__(image_shape=image_shape,
                                                 labels_path=labels_path)
        self.model = torchvision.models.inception_v3(pretrained=True, transform_input=True)
        self.model_name = 'InceptionV3_public'

    def forward(self, x):
        return self.model.forward(x)

    def get_cutted_model(self, bottleneck):
        return InceptionV3_cutted(self.model, bottleneck)


class SmallResNet50Wrapper(PublicImageModelWrapper):

    def __init__(self, labels_path):
        image_shape = [299, 299, 3]
        super(SmallResNet50Wrapper, self).__init__(image_shape=image_shape,
                                                  labels_path=labels_path)
        self.model = torchvision.models.__dict__["resnet50"](pretrained=True)
        self.model_name = 'ResNet50_8class_public'

    def forward(self, x):
        return self.model.forward(x)

    def train(self, train_loader, criterion, optimizer, epoch, lists, load_weights_path="D:\\torch-model-weights\\resnet50_8class.pt"):
        if load_weights_path:
            self.model.load_state_dict(torch.load(load_weights_path))
            self.model.eval()
        else:
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')

            progress = ProgressMeter(
                len(train_loader),
                [batch_time, data_time, losses, top1, top5],
                prefix="Epoch: [{}]".format(epoch))

            end = time.time()
            self.model.train()

            for i, (train_data, target) in enumerate(train_loader):
                data_time.update(time.time() - end)
                train_data = train_data.to(device)
                target = target.to(device)

                output = self.model(train_data)
                loss = criterion(output, target)
                acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
                print(output)
                losses.update(loss.item(), train_data.size(0))
                top1.update(acc1[0], train_data.size(0))
                top5.update(acc5[0], train_data.size(0))
                lists[0].append(loss.item())
                # print('lst_loss: ', lists[0])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_time.update(time.time() - end)
                end = time.time()

                if i % printing_freq == 0:
                    progress.display(i)

    def validate(self, val_loader, criterion, lists):
        batch_time = AverageMeter('Time', '6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix='Test: ')

        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (val_data, target) in enumerate(val_loader):
                val_data = val_data.to(device)
                target = target.to(device)

                output = self.model(val_data)
                loss = criterion(output, target)

                acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), val_data.size(0))
                top1.update(acc1[0], target.size(0))
                top5.update(acc5[0], target.size(0))
                batch_time.update(time.time() - end, target.size(0))
                end = time.time()

            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
            lists[1].append(top1.avg)
            lists[2].append(top5.avg)
            print('lst_acc1: ', lists[1])
            print('lst_acc5: ', lists[2])

        return top1.avg

    def get_cutted_model(self, bottleneck, is_trained=True):
        if is_trained:
            return SmallResNet50_cutted(self.model, bottleneck)
        else:
            data_dir = "D:/natural_images_dataset/natural_images"
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            full_dataset = datasets.ImageFolder(data_dir, transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                normalize
            ]))

            dataset_size = len(full_dataset)
            indices = list(range(dataset_size))
            np.random.shuffle(indices)

            train_split = int(np.floor(0.8 * dataset_size))
            train_indices, other_indices = indices[:train_split], indices[train_split:]

            other_indices_size = len(other_indices)
            test_split = int(np.floor(0.5 * other_indices_size))
            test_indices, val_indices = other_indices[:test_split], other_indices[test_split:]

            # Defining samplers for future loader use
            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(test_indices)
            val_sampler = SubsetRandomSampler(val_indices)

            # Some hyperparameters
            batch_size = 50
            num_workers = 2
            lr = 0.01
            momentum = 0.9
            weight_decay = 5e-4
            epochs = 10

            # Need to apply data augmentation on train set split ......
            train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=num_workers)  # Augmentation
            val_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False, num_workers=num_workers)
            test_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=num_workers)

            # Constructing the model & setting up
            print('designated device: ', device)
            self.model = self.model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(self.model.parameters(), lr, momentum=momentum)

            self.main(epochs, self.model, train_loader, val_loader, test_loader, criterion, optimizer)
            return SmallResNet50_cutted(self.model, bottleneck)

    def main(self, epochs, train_loader, val_loader, test_loader, criterion, optimizer):
        best_acc = 0.0
        best_epoch = 0
        lst_loss = []
        lst_acc_1 = []
        lst_acc_2 = []
        lists = (lst_loss, lst_acc_1, lst_acc_2)
        for epoch in range(0, epochs):
            self.train(self.model, train_loader, criterion, optimizer, epoch, lists)
            acc1 = self.validate(val_loader, criterion, lists)
            if acc1 > best_acc:
                best_acc = acc1
                best_epoch = epoch+1

        print('Best validation is from epoch ', best_epoch, ', with accuracy of ', best_acc.item())
        final_acc = self.validate(test_loader, criterion, lists)
        print('Final test set result: ', final_acc)

    @staticmethod
    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_num = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_num))
            return res


class SmallResNet50_cutted(torch.nn.Module):
    def __init__(self, resnet50_8class, bottleneck):
        super(SmallResNet50_cutted, self).__init__()
        names = list(resnet50_8class._modules.keys())
        layers = list(resnet50_8class.children())

        self.layers = torch.nn.ModuleList()
        self.layers_names = []

        bottleneck_met = False

        for name, layer in zip(names, layers):
            if name == bottleneck:
                bottleneck_met = True
                continue
            if not bottleneck_met:
                continue

            self.layers.append(layer)
            self.layers_names.append(name)

    def forward(self, x):
        output = x
        for i in range(0, len(self.layers)):
            if self.layers_names[i] == 'fc':
                # flatten: tensor with dimension (2,4,3,5,6) becomes (2, 4*3*5, 6) after calling flatten(t, 1, 3)
                output = torch.flatten(output, 1)
            output = self.layers[i](output)
        return output



