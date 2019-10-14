# ref: imagenet example:
#   https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py
# torchvision/models document:
#   https://pytorch.org/docs/stable/torchvision/models.html

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='Pytorch Resnet50')
parser.add_argument('--data', metavar='DIR',
                    help='path to the image net dataset',
                    default='/mnt/sda/imagenet2012')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loadign workers(default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size(default: 256)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--loop', '-l', default=100, type=int,
                    metavar='N', help='loop many batches before exit(default: 100)')


image_size = 224


class ModelData(object):
    INPUT_SHAPE = (3, image_size, image_size)


def validate(val_loader, model):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            target = target.cuda(async=True)
            # input_var = torch.autograd.Variable(input, volatile=True)
            # target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = model(input.cuda(async=True))
            # loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            # losses.update(loss.data[0], input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          top1=top1, top5=top5))
            if i == args.loop:
                break

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def main():
    global args
    args = parser.parse_args()

    # load model
    resnet50 = models.resnet50(pretrained=True)
    resnet50.cuda()

    # data loading
    #
    # All pre-trained models expect input images normalized in the same way,
    # i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and
    # W are expected to be at least 224. The images have to be loaded in to a
    # range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and
    # std = [0.229, 0.224, 0.225]

    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # imagenet_data = datasets.ImageNet(
    #     args.data, split='train', transform=transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]),
    #     download=False
    # )

    # print("size of Imagenet data is {}".format(len(imagenet_data)))

    # data_loader = torch.utils.data.DataLoader(
    #     imagenet_data,
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)

    # define loss function (criterion)
    # criterion = nn.CrossEntropyLoss().cuda()

    # validate(data_loader, resnet50)
    run(resnet50)

    return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def run(model):
    batch_time = AverageMeter()
    end = time.time()
    input = torch.rand((args.batch_size,) + ModelData.INPUT_SHAPE)
    with torch.no_grad():
        for i in range(args.loop):
            input_cuda = input.cuda(non_blocking=True)
            model(input_cuda)
            # torch.cuda.synchronize()
            batch_time.update(time.time() - end, end)
            end = time.time()
            if i % args.print_freq == 0:
                print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                    batch_time=batch_time))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
