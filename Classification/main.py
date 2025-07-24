# !/usr/bin/env python3

"""
Demonstrates how to:
    * use the MAML wrapper for fast-adaptation,
    * use the benchmark interface to load mini-ImageNet, and
    * sample tasks and split them in adaptation and evaluation sets.
To contrast the use of the benchmark interface with directly instantiating mini-ImageNet datasets and tasks, compare with `protonet_miniimagenet.py`.
"""

import random
import numpy as np
import timm
import os
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import json
 
import learn2learn as l2l
from models import *
import timm
import argparse


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def get_args_parser():
    parser = argparse.ArgumentParser('Parameters Setting', add_help=False)
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--ways', default=5)
    parser.add_argument('--shots', default=5)
    parser.add_argument('--meta_lr', default=1e-3)
    parser.add_argument('--fast_lr', default=1e-2)
    parser.add_argument('--meta_batch_size', default=64)
    parser.add_argument('--adaptation_steps', default=5)
    parser.add_argument('--num_iterations', default=15000)
    parser.add_argument('--seed', default=42)
    parser.add_argument('--model_name', default='resnet50')
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--save_dir', default='./SavedModels')
    parser.add_argument('--mask_rate', default=0.5)
    parser.add_argument('--mask_layer', default=10)
    args = parser.parse_args()
    return args

def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(adaptation_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    evaluation_error = loss(predictions, evaluation_labels)
    evaluation_accuracy = accuracy(predictions, evaluation_labels)
    return evaluation_error, evaluation_accuracy


def main(args):
    args.log_dir = os.path.join(args.log_dir, args.model_name)
    args.save_dir = os.path.join(args.save_dir, args.model_name+'.pt')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    writer = SummaryWriter(args.log_dir)

    # Create Tasksets using the benchmark interface
    tasksets = l2l.vision.benchmarks.get_tasksets('mini-imagenet',
                                                  train_samples=2 * args.shots,
                                                  train_ways=args.ways,
                                                  test_samples=2 * args.shots,
                                                  test_ways=args.ways,
                                                  root='./data',
                                                  data_augmentation='lee2019'
                                                  )
    if 'mask' in args.model_name:
        model = timm.create_model(args.model_name, num_classes=args.ways,
                                  mask_rate=args.mask_rate, mask_layer=args.mask_layer)
    else:
        model = timm.create_model(args.model_name, num_classes=args.ways)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=args.fast_lr)
    opt = optim.Adam(maml.parameters(), args.meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')
    args_dict = vars(args)
    args_dict['TestError'] = 0.0
    args_dict['TestAccuracy'] = 0.0
    for iteration in range(args.num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(args.meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch = tasksets.train.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               args.adaptation_steps,
                                                               args.shots,
                                                               args.ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            batch = tasksets.validation.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               args.adaptation_steps,
                                                               args.shots,
                                                               args.ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

        # Print some metrics
        writer.add_scalar('Meta Train/Error', meta_train_error / args.meta_batch_size, iteration)
        writer.add_scalar('Meta Train/Accuracy', meta_train_accuracy / args.meta_batch_size, iteration)
        writer.add_scalar('Meta Valid/Error', meta_valid_error / args.meta_batch_size, iteration)
        writer.add_scalar('Meta Valid/Accuracy', meta_valid_accuracy / args.meta_batch_size, iteration)
        print('Iteration', iteration, end=' ')
        print('Train Accuracy:{:.3f}'.format(meta_train_accuracy / args.meta_batch_size), end=' ')
        print('Meta Valid Error:{:.3f}'.format(meta_valid_error / args.meta_batch_size), end=' ')
        print('Accuracy:{:.3f}'.format(meta_valid_accuracy / args.meta_batch_size))
        if (iteration+1)%100 == 0:
            torch.save(model.state_dict(), args.save_dir)
            print('Save Model: ', args.save_dir)
        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / args.meta_batch_size)
        opt.step()
        if (iteration+1) % 500 == 0:
            meta_test_error = 0.0
            meta_test_accuracy = 0.0
            for task in range(args.meta_batch_size):
                # Compute meta-testing loss
                learner = maml.clone()
                batch = tasksets.test.sample()
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   learner,
                                                                   loss,
                                                                   args.adaptation_steps,
                                                                   args.shots,
                                                                   args.ways,
                                                                   device)
                meta_test_error += evaluation_error.item()
                meta_test_accuracy += evaluation_accuracy.item()
            print('Meta Test Error', meta_test_error / args.meta_batch_size)
            print('Meta Test Accuracy', meta_test_accuracy / args.meta_batch_size)
            writer.add_scalar('Meta Test/Error', meta_test_error / args.meta_batch_size, (iteration+1)//500)
            writer.add_scalar('Meta Test/Accuracy', meta_test_accuracy / args.meta_batch_size, (iteration+1)//500)
            if args_dict['TestAccuracy'] < meta_test_accuracy / args.meta_batch_size:
                args_dict['TestError'] = meta_test_error / args.meta_batch_size
                args_dict['TestAccuracy'] = meta_test_accuracy / args.meta_batch_size
                torch.save(model.state_dict(), args.save_dir)
                with open('results/'+args.model_name+".json", "w") as f:
                    f.write(json.dumps(args_dict, ensure_ascii=False, indent=4, separators=(',', ':')))

if __name__ == '__main__':
    args = get_args_parser()
    main(args)
