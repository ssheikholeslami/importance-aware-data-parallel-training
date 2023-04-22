##############################
#
# Code for "The Impact of Importance-aware Dataset Partitioning on Data-parallel Training of Deep Neural Networks".
# Authors: Sina Sheikholeslami, Amir H. Payberah, Tianze Wang, Jim Dowling, Vladimir Vlassov
# Publication Date: April 2023
# In case of questions open an issue on https://github.com/ssheikholeslami/policy-based-dataset-partitioning
# or contact sinash@kth.se
#
#############################
import os
from datetime import datetime, timedelta
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import time
import operator

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.tensorboard import SummaryWriter

from torchinfo import summary


from samplers import importance_sampler
from utils import datasetutils, exampleutils
import importance


########### PROCESS SETUP & MAIN #######################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--total-epochs', type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--exid', default=0, type=int,
                        help='experiment id, used for, e.g., prefixing the directory for saving the examples_losses '
                             'matrices')
    parser.add_argument('--warmup-epochs', type=int,
                        help='number of warmup training rounds before dataset redistribution')
    parser.add_argument('--seed', type=int,
                        help='random seed for PyTorch')
    parser.add_argument('--heuristic', type=str,
                        help='heuristic for repartitioning after warmup')
    parser.add_argument('--measure', type=str,
                        help='measure (e.g., average, variance)')
    parser.add_argument('--batch-size', type=int,
                        help='batch size for training')
    parser.add_argument('--shuffle', type=str,
                        help='after warmup, whether to shuffle example orders at each epoch or not, no/yes')
    parser.add_argument('--dataset', type=str,
                        help='name of the dataset to be used for the experiment')
    parser.add_argument('--model', type=str,
                        help='name of the model to be used for the experiment')
    parser.add_argument('--description', default='', type=str,
                        help='short description of the experiment')
    parser.add_argument('--once-or-interval', default='once', type=str,
                        help='indicates whether the repartitioning happens once or happens at fixed intervals')
    parser.add_argument('--interval-epochs', default=0, type=int,
                        help='interval (epochs) for dataset repartitioning')
    parser.add_argument('--ignore-epochs', default=0, type=int,
                        help='number of initial epochs to ignore for example importance calculation')
    args = parser.parse_args()

    #########################################################

    if args.once_or_interval.lower() == 'interval':
        if not(args.interval_epochs > 0):
            raise RuntimeError(
                f"You've indicated interval repartitioning, but the interval epochs is {args.interval_epochs}."
            )
        if args.heuristic.lower() == 'none':
            raise RuntimeError(
                f"You've indicated interval repartitioning, but selected the 'none' heuristic as well."
            )
    elif args.once_or_interval.lower() == 'once':
        if args.interval_epochs != 0:
            raise RuntimeError(
                f"You've indicated one attempt at repartitioning (once), but the interval_epochs is not 0."
            )
        args.interval_epochs = args.total_epochs - args.warmup_epochs
    
    if args.warmup_epochs > args.total_epochs:
        raise RuntimeError(
            "Number of total epochs should be greater than or equal to the number of warmup epochs."
        )
    if args.dataset is None:
        raise RuntimeError(
            "Dataset is not specified."
        )

    args.world_size = args.gpus * args.nodes  #
    os.environ['MASTER_ADDR'] = 'localhost'  #
    os.environ['MASTER_PORT'] = '12345'  #
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging (OFF, INFO, DETAIL). DETAIL may cause hanging behavior

    experiment_name = f"{args.description}###T{args.total_epochs}-W{args.warmup_epochs}-IG{args.ignore_epochs}-INT{args.interval_epochs}- MEAS'{args.measure}' - H'{args.heuristic}' - {args.dataset} - {args.model} - Shuffle {args.shuffle} - Seed {args.seed} + B{args.batch_size} - O/I:{args.once_or_interval}" + f" - {time.asctime()}"

    # directory to save worker matrices
    save_dir = os.getcwd() + '/experiment_results/' + experiment_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # warmup training uses vanilla DDP, i.e., no heuristic
    # so in case there are no heuristics used later, the warmup is the full training
    if args.heuristic.lower() == 'none':
        args.warmup_epochs = args.total_epochs

    mp.spawn(train, nprocs=args.gpus, args=(experiment_name, args,))  #
    #########################################################


def train(gpu, experiment_name, args):

    if args.shuffle.lower() == 'no':
        epoch_shuffle = False
    elif args.shuffle.lower() == 'yes':
        epoch_shuffle = True
    else:
        raise RuntimeError(
            "Heuristic training epoch shuffle behavior (--shuffle) must be either 'yes' or 'no'."
        )

    ###### TensorBoard for GPU 0 #########
    if gpu == 0:
        import wandb

        tensorboard_writer = SummaryWriter(comment=" " + experiment_name)
        
        # UNCOMMENT FOR WANDB
        # wandb.init(project="YOUR_PROJECT",
        #            entity="YOUR_ENTITY", name=experiment_name)
        # wandb.config.update(args)

    ############################################################
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )

    group_gloo = dist.new_group(backend="gloo")
    # print(f"Rank is {rank} and device id is {gpu}")

    ############################################################
    torch.manual_seed(args.seed)

    # preliminary dataset setup, use num_classes for model init
    if args.dataset.lower() == 'cifar10':
        ExperimentDataset = datasetutils.dataset_with_indices(
            torchvision.datasets.CIFAR10)
        num_classes = 10
        num_channels = 3
    elif args.dataset.lower() == 'cifar100':
        ExperimentDataset = datasetutils.dataset_with_indices(
            torchvision.datasets.CIFAR100)
        num_classes = 100
        num_channels = 3
    elif args.dataset.lower() == 'fashionmnist':
        ExperimentDataset = datasetutils.dataset_with_indices(
            torchvision.datasets.FashionMNIST)
        num_classes = 10
        num_channels = 1
    # ADD/EXTEND OTHER DATASETS HERE
    else:
        raise RuntimeError(
            "Specified dataset is not supported or cannot be found. Make sure you pass a correct --dataset runtime argument. \n The experiment will be terminated."
        )

    if args.model.lower() == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False)
        # for Fashion-MNIST, change the input features
        if num_channels == 1:
            # model.conv1.in_channels = 1
            model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7,
                  stride=2, padding=3, bias=False)
        # change output layer
        model.fc = nn.Linear(512, num_classes)
    elif args.model.lower() == 'resnet34':
        model = torchvision.models.resnet34(pretrained=False)
        # for Fashion-MNIST, change the input features
        if num_channels == 1:
            model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7,
                                    stride=2, padding=3, bias=False)        
        # change output layer
        model.fc = nn.Linear(512, num_classes)
    elif args.model.lower() == 'resnet50':
        model = torchvision.models.resnet50(pretrained=False)
        # for Fashion-MNIST, change the input features
        if num_channels == 1:
            model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7,
                                    stride=2, padding=3, bias=False)
        # change output layer
        model.fc = nn.Linear(2048, num_classes)
    elif args.model.lower() == 'vgg16':
        model = torchvision.models.vgg16(pretrained=False)
        # for Fashion-MNIST, change the input features
        # if num_channels == 1:
        #     model.features[0] = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1)
        # change output layer
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif args.model.lower() == 'alexnet':
        model = torchvision.models.alexnet(pretrained=False)
        # for Fashion-MNIST, change the input features
        # if num_channels == 1:
        #     model.features[0] = nn.Conv2d(num_channels, 64, kernel_size=11, stride=4, padding=2)
        # change output layer
        model.classifier[6] = nn.Linear(4096, num_classes)
    # ADD/EXTEND OTHER MODELS HERE
    else:
        raise RuntimeError(
            "Specified model is not supported or cannot be found. Make sure you pass a correct --model runtime argument. \n The experiment will be terminated."
        )

    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = args.batch_size

    # define loss function (criterion) and optimizer
    criterion_individual = nn.CrossEntropyLoss(reduction='none').cuda(gpu)
    criterion_mean = nn.CrossEntropyLoss().cuda(gpu)
    # optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.1,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=5e-4,
                                )

    ###############################################################
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])
    ###############################################################

    # Transforms
    if args.dataset.lower() in ['cifar10', 'cifar100'] :
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)),
        ])
    
    # Fashion-MNIST is greyscale
    # values taken from https://github.com/LucasVandroux/Fashion-MNIST-PyTorch/blob/master/train_fashionMNIST.py
    elif args.dataset.lower() == 'fashionmnist':
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           (0.1307,), (0.3081,)),
        ])
    # Data loading code

    train_dataset = ExperimentDataset(
        root='./data', train=True, download=True, transform=transform)
    ########################### SAMPLER - WARMUP ############################
    # for args.warmup_epochs, use the DistributedSampler (the default distributed random sampler)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank,
        shuffle=True
    )
    ######################### DATA LOADERS - WARMUP ##########################

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               # shuffle=False,  # TODO documentation says that shuffle XOR sampler should be provided
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               )

    test_dataset = ExperimentDataset(
        root='./data', train=False, download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              )

    num_examples = train_dataset.__len__()  # cifar10 : 50000

    start_time = datetime.now()
    # initialize the local example-loss matrix
    examples_losses = torch.zeros([num_examples, args.warmup_epochs]).cuda(gpu)
    # print(examples_losses, "GPU: ", gpu)

    total_steps = len(train_loader)

    ################################ WARMUP TRAINING ####################################
    current_overall_epoch = 0

    for epoch in range(args.warmup_epochs):
        cumulative_overhead = timedelta()
        train_sampler.set_epoch(epoch)
        # train_heuristic_sampler.set_epoch(epoch)
        # print("EPOCH: ", epoch)
        for i, data in enumerate(train_loader, 0):

            # data is a list of [images, labels, indices]
            images, labels, indices = data[0].cuda(non_blocking=True), \
                data[1].cuda(non_blocking=True), \
                data[2].cuda(non_blocking=True)

            size_of_this_batch = len(data[0])
            # check epoch shuffle behavior
            # if i == 0 and gpu == 0:
            #     print(f"epoch {epoch} warmup step 0 gpu 0")

            # Forward pass
            outputs = model(images)
            if gpu == 0:
                overhead_start_time=datetime.now()
            ############################### UPDATE THE MATRIX #################################
            individual_losses = criterion_individual(outputs, labels)
            # update the examples_losses matrix with loss values from this mini-batch
            for ex_in_batch in range(size_of_this_batch):
                examples_losses[indices[ex_in_batch].item(
                )][epoch] = individual_losses[ex_in_batch].item()
                # uncomment below for testing the placements
                if indices[ex_in_batch] == 1:
                    print("Index 1 Loss Update:\n")
                    print(examples_losses[indices[ex_in_batch].item()])
            ###################################################################################
            if gpu == 0:
                cumulative_overhead += datetime.now()-overhead_start_time
            # continue training as usual

            loss = criterion_mean(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
               
        current_overall_epoch+=1


        if gpu == 0:
            # UNCOMMENT FOR WANDB
            # wandb.log({'importance_tracking_overhead': cumulative_overhead.total_seconds()}, step=epoch + 1)
            print(f'Epoch {epoch + 1} finished.')
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.total_epochs,
                                                       loss.item()))

            tensorboard_writer.add_scalar("Loss/train", loss.item(), epoch+1)
            # UNCOMMENT FOR WANDB
            # wandb.log({'train_loss': loss}, step=epoch + 1)

            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data[0].to(gpu), data[1].to(gpu)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    accuracy = 100 * correct / total
            print(f"Test Accuracy: {accuracy:.2f}%")
            tensorboard_writer.add_scalar("Accuracy/test", accuracy, epoch+1)
            tensorboard_writer.flush()
            # UNCOMMENT FOR WANDB
            # wandb.log({'test_accuracy': accuracy}, step=epoch + 1)

    if gpu == 0:
        print("Training stage completed in: " +
              str(datetime.now() - start_time))

    # end training here unless heuristic != 'none'
    if args.heuristic.lower() == 'none':
        if gpu == 0:
            tensorboard_writer.close()
            # UNCOMMENT FOR WANDB
            # wandb.finish()
        return

    # RESUME WITH HEURISTIC-BASED TRAINING
    if gpu == 0:
        overhead_start_time = datetime.now()
    num_periods = int((args.total_epochs - args.warmup_epochs) / args.interval_epochs)
    final_period_epochs = ((args.total_epochs - args.warmup_epochs) % args.interval_epochs) + args.interval_epochs

    for current_period in range(num_periods):
        # account for last periods leftover epochs
        if current_period == num_periods - 1:
            args.interval_epochs = final_period_epochs
        if gpu == 0:
            print(f"Proceeding with period {current_period+1} for {args.interval_epochs} epochs")
        # prepare new allocations
        examples_to_workers = prepare_allocations(
            examples_losses, rank, group_gloo, current_overall_epoch, experiment_name, args)
        examples_losses = torch.zeros([num_examples, args.interval_epochs]).cuda(gpu)
        train_sampler, train_loader = reallocate_examples(train_dataset=train_dataset,
                                                        examples_allocations=examples_to_workers,
                                                        batch_size=batch_size,
                                                        num_replicas=args.world_size,
                                                        num_workers=0,
                                                        shuffle=epoch_shuffle,
                                                        rank=rank,
                                                        seed=args.seed,
                                                        )
        if gpu == 0:
            heuristic_partitioning_overhead = datetime.now() - overhead_start_time
            # UNCOMMENT FOR WANDB
            # wandb.log({'heuristic_overhead': heuristic_partitioning_overhead.total_seconds()})

        # Resume training with new partitions
        starting_epoch = current_overall_epoch
        for epoch in range(starting_epoch, starting_epoch+args.interval_epochs):
        # for epoch in range(args.warmup_epochs, args.total_epochs):
            cumulative_overhead = timedelta()
            train_sampler.set_epoch(epoch)
            for i, data in enumerate(train_loader, 0):
                # data is a list of [images, labels, indices]
                images, labels, indices = data[0].cuda(non_blocking=True), \
                    data[1].cuda(non_blocking=True), \
                    data[2].cuda(non_blocking=True)

                size_of_this_batch = len(data[0])
                # check epoch shuffle behavior
                # if i == 0 and gpu == 0:
                #     print(f"epoch {epoch} heuristic step 0 gpu 0")
                # Forward pass
                outputs = model(images)
                if gpu == 0:
                    overhead_start_time = datetime.now()
                # individual loss and matrix calculation code goes here
                ############################### UPDATE THE MATRIX #################################
                individual_losses = criterion_individual(outputs, labels)
                # update the examples_losses matrix with loss values from this mini-batch
                for ex_in_batch in range(size_of_this_batch):
                    examples_losses[indices[ex_in_batch].item(
                    )][epoch-starting_epoch] = individual_losses[ex_in_batch].item()
                    # uncomment below for testing the placements
                    if indices[ex_in_batch] == 1:
                        print("Index 1 Loss Update:\n")
                        print(examples_losses[indices[ex_in_batch].item()])
                if gpu == 0:
                    cumulative_overhead += datetime.now()-overhead_start_time
                # continue training as usual

                loss = criterion_mean(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            current_overall_epoch +=1

            if gpu == 0:
                # UNCOMMENT FOR WANDB
                # wandb.log({'importance_tracking_overhead': cumulative_overhead.total_seconds()}, step=epoch + 1)
                print(f'Epoch {epoch + 1} finished.')
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.total_epochs,
                                                        loss.item()))
                tensorboard_writer.add_scalar("Loss/train", loss.item(), epoch+1)
                # UNCOMMENT FOR WANDB
                # wandb.log({'train_loss': loss}, step=epoch + 1)
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in test_loader:
                        images, labels = data[0].to(gpu), data[1].to(gpu)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        accuracy = 100 * correct / total
                print(f"Test Accuracy: {accuracy:.2f}%")
                tensorboard_writer.add_scalar("Accuracy/test", accuracy, epoch+1)
                tensorboard_writer.flush()
                # UNCOMMENT FOR WANDB
                # wandb.log({'test_accuracy': accuracy}, step=epoch + 1)



    if gpu == 0:
        tensorboard_writer.close()
        # UNCOMMENT FOR WANDB
        # wandb.finish()


if __name__ == '__main__':
    main()


def prepare_allocations(examples_losses, rank, group_gloo, current_overall_epoch, experiment_name, args):
    save_dir = os.getcwd() + '/experiment_results/' + experiment_name
    file_name = save_dir + '/' + 'Rank ' + str(rank) + '- examples_losses.pt'
    torch.save(examples_losses, file_name)

    # wait for everyone to finish
    # can this cause hanging because of incomplete collective communication?
    torch.distributed.monitored_barrier(group=group_gloo, wait_all_ranks=True)

    #  merge all partial matrices on rank 0
    if rank == 0 and args.heuristic != 'none':

        base_matrix_path = save_dir + '/Rank 0- examples_losses.pt'
        examples_losses = exampleutils.merge_worker_matrices(
            args.world_size, save_dir, base_matrix_path=base_matrix_path)

        print("merging of matrices completed, examples_losses: ", examples_losses)

        ############################## CALCULATE IMPORTANCE ##################################
        print("Calculating importance of examples based on loss values.")
        # ignore first few epochs during warmup
        if current_overall_epoch == args.warmup_epochs:
            if args.ignore_epochs > 0:
            # drop the ignore_epochs first columns
                examples_losses = examples_losses[:, args.ignore_epochs:]
        # sort examples by their average loss, in ascending order
        sorted_examples = sorted(importance.calculate_metric_measure(
            examples_losses, args.measure).items(), key=lambda x: x[1], reverse=True)

        num_examples = len(sorted_examples)
        ############################## CALCULATE ALLOCATIONS ##################################
        print(
            f"Repartitioning the dataset over workers based on {args.heuristic} policy")
        sorted_indices = list(map(operator.itemgetter(0), sorted_examples))
        examples_to_workers = {worker: []
                               for worker in range(args.world_size)}

        if args.heuristic.lower() == 'roundrobin':
            for index, example in enumerate(sorted_indices):
                examples_to_workers[index % args.world_size].append(example)
        elif args.heuristic.lower() == 'steps':
            # give the top N/W examples of the sorted list to W1, the next N/W to W2, etc.
            num_examples_per_worker = int(num_examples/args.world_size)
            for worker in range(args.world_size):
                examples_to_workers[worker] = sorted_indices[worker *
                                                             num_examples_per_worker:(worker+1)*num_examples_per_worker]
        # ADD/EXTEND OTHER HEURISTICS HERE

        # write allocations dict to file so other processes have it
        torch.save(examples_to_workers, save_dir + '/allocations_dictionary')
        print("Wrote the allocations dictionary to disk")

        torch.distributed.monitored_barrier(
            group=group_gloo, wait_all_ranks=True)
    else:
        # others wait for rank 0 to finish merging and calculation of importances and allocations
        torch.distributed.monitored_barrier(
            group=group_gloo, wait_all_ranks=True)

    if args.heuristic.lower() != 'none':
        examples_to_workers = torch.load(save_dir + '/allocations_dictionary')
        torch.distributed.monitored_barrier(
            group=group_gloo, wait_all_ranks=True)

    return examples_to_workers


def reallocate_examples(train_dataset, examples_allocations, batch_size,
                        num_replicas, num_workers, shuffle, rank, seed,
                        ):

    train_sampler = importance_sampler.ConstantSampler(
        train_dataset,
        num_replicas=num_replicas,
        examples_allocations=examples_allocations,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
    )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               # shuffle=False,  # TODO documentation says that shuffle XOR sampler should be provided
                                               num_workers=num_workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               )

    return train_sampler, train_loader