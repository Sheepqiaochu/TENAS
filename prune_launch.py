import os
import time
import argparse

# TODO please configure TORCH_HOME and data_paths before running
TORCH_HOME = "/data/users/yangqiancheng/datasets/"
# Path that contains the nas-bench-201 database. If you only want to run on NASNET (i.e.
# DARTS) search space, then just leave it empty
data_paths = {
    "cifar10": "/data/users/yangqiancheng/datasets/",
    "cifar100": "/data/users/yangqiancheng/datasets/",
    "ImageNet16-120": "/data/users/yangqiancheng/datasets/ImageNet16",
    "imagenet-1k": "/data/users/yangqiancheng/datasets/imagenet-data",
}

model_paths = {
    "cifar10_resnet56": "/data/users/yangqiancheng/models/UAP/cifar10_resnet56.pth.tar",
    "cifar10_vgg16": "/data/users/yangqiancheng/models/UAP/cifar10_vgg16.pth.tar",
    "cifar10_vgg19":"/data/users/yangqiancheng/models/UAP/cifar10_vgg19.pth.tar",
    "cifar100_resnet56":"/data/users/yangqiancheng/models/UAP/cifar100_resnet56.pth.tar",
    "cifar100_vgg16": "/data/users/yangqiancheng/models/UAP/cifar109_vgg16.pth.tar",
    "cifar100_vgg19":"/data/users/yangqiancheng/models/UAP/cifar100_vgg19.pth.tar",
}


# This parser is used to add specify some settings from terminal
parser = argparse.ArgumentParser("TENAS_launch")
parser.add_argument('--gpu', type=int,   help='use gpu with cuda number')
parser.add_argument('--space', default='nas-bench-201', type=str, choices=['nas-bench-201', 'darts'],
                    help='which nas search space to use')
parser.add_argument('--dataset', default='cifar100', type=str,
                    choices=['cifar10', 'cifar100', 'ImageNet16-120', 'imagenet-1k'],
                    help='Choose between cifar10/100/ImageNet16-120/imagenet-1k')
parser.add_argument('--seed', default=0, type=int, help='manual seed')
parser.add_argument('--UAP_info', type=str, help='info 0of UAP models')

args = parser.parse_args()

# Basic Settings
precision = 3
# init = 'normal'
# init = 'kaiming_uniform'
init = 'kaiming_normal'

if args.space == "nas-bench-201":
    prune_number = 1
    batch_size = 72
    space = "nas-bench-201"  # different spaces of operator candidates, not structure of supernet
    super_type = "basic"  # type of supernet structure
elif args.space == "darts":
    space = "darts"
    super_type = "nasnet-super"
    if args.dataset == "cifar10":
        prune_number = 3
        batch_size = 14
        # batch_size = 6
    elif args.dataset == "imagenet-1k":
        prune_number = 2
        batch_size = 24

# if args.UAP_info == 'cifar10_resnet56':
#     UAP_generator = '/data/users/yangqiancheng/experiment_results/UAP_generator/results/cifar10_untargeted/2021-11-26_04:11:22_cifar10_resnet56_cifar10_123/checkpoint.pth.tar'
# elif args.UAp_generator == 'cifar10_vgg19':



timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.gmtime(time.time())))
# UAP_generator = '/data/users/yangqiancheng/models/UAP/imagenet32_resnet152.pth.tar'



core_cmd = "CUDA_VISIBLE_DEVICES={gpuid} OMP_NUM_THREADS=4 python ./prune_tenas.py \
--save_dir {save_dir} --max_nodes {max_nodes} \
--dataset {dataset} \
--data_path {data_path} \
--search_space_name {space} \
--super_type {super_type} \
--arch_nas_dataset {TORCH_HOME}/NAS-Bench-201-v1_1-096897.pth \
--track_running_stats 1 \
--workers 0 --rand_seed {seed} \
--timestamp {timestamp} \
--precision {precision} \
--init {init} \
--repeat 3 \
--batch_size {batch_size} \
--prune_number {prune_number} \
--UAP_generator {UAP_generator}       \
".format(
    gpuid=args.gpu,
    save_dir="./output/prune-{space}/{dataset}{UAP_info}".format(space=space,\
         dataset=args.dataset,UAP_info=args.UAP_info),
    max_nodes=4,
    data_path=data_paths[args.dataset],
    dataset=args.dataset,
    TORCH_HOME=TORCH_HOME,
    space=space,
    super_type=super_type,
    seed=args.seed,
    timestamp=timestamp,
    precision=precision,
    init=init,
    batch_size=batch_size,
    prune_number=prune_number,
    UAP_generator=model_paths[args.UAP_info]
)

os.system(core_cmd)
