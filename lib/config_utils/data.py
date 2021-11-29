def get_data_specs(pretrained_dataset):
    if pretrained_dataset == "imagenet-1k":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        num_classes = 1000
        input_size = 32
        # input_size = 299 # inception_v3
        num_channels = 3
    elif pretrained_dataset == "ImageNet16-120":
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22, 61.26, 65.09]]
        num_classes = 100
        input_size = 32
        num_channels = 3
    elif pretrained_dataset == "cifar10":
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        num_classes = 10
        input_size = 32
        num_channels = 3
    elif pretrained_dataset == "cifar100":
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
        num_classes = 100
        input_size = 32
        num_channels = 3
    else:
        raise ValueError
    return num_classes, (mean, std), input_size, num_channels
