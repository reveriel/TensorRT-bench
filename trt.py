# ref: imagenet example:
#   https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py
# torchvision/models document:
#   https://pytorch.org/docs/stable/torchvision/models.html


# bug,
# [TensorRT] ERROR: ../rtSafe/cuda/caskConvolutionRunner.cpp (245) - Cask Error in checkCaskExecError<false>: 7 (Cask Convolution execution)
# https://devtalk.nvidia.com/default/topic/1056268/tensorrt/tensorrt-do_inference-error/

# use tensorrt to run resnet50

import argparse
import os
import time
import common

import random
from PIL import Image
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# pip install nvidia-ml-py3
import nvidia_smi
nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # card 0


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
parser.add_argument('-q', action='store_true',
                    help="use int8 quantization")

parser.add_argument('--csv', action='store_true', help="csv output")

# You can set the logger severity higher to suppress messages (or lower to display more messages).
# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_LOGGER = trt.Logger(trt.Logger.INTERNAL_ERROR)
# TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

image_size = 224


class ModelData(object):
    # MODEL_PATH = "resnet50.onnx"   # generated by myself, doesn't work
    MODEL_PATH = "ResNet50.onnx"  # given by nvidia
    INPUT_SHAPE = (3, image_size, image_size)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32


# Allocate host and device buffers, and create a stream.
def allocate_buffers(engine):
    # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
    h_input = cuda.pagelocked_empty(trt.volume(
        engine.get_binding_shape(0)) * args.batch_size, dtype=trt.nptype(ModelData.DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(
        engine.get_binding_shape(1)) * args.batch_size, dtype=trt.nptype(ModelData.DTYPE))
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream


def do_inference(context, h_input, d_input, h_output, d_output, stream):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    context.execute_async(bindings=[int(d_input), int(
        d_output)], stream_handle=stream.handle, batch_size=args.batch_size)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()


# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = common.GiB(1)
        builder.max_batch_size = args.batch_size

        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            ok = parser.parse(model.read())
            if not ok:
                print("Error: Parse onnx model \"{}\" failed.".format(model_file))
                error = parser.get_error(0)
                print("  code: {}".format(error.code()))
                print("  desc: {}".format(error.desc()))
                print("  file: {}".format(error.file()))
                print("  func: {}".format(error.func()))
                print("  line: {}".format(error.line()))
                print("  node: {}".format(error.node()))
                exit(-1)

        if args.q:
           # enable int8 and set quantize (chenrong06)
            # Incomplete version, please refer to workspace/tensorrt/samples/sampleINT8API/sampleINT8API.cpp
            builder.int8_mode = True
            builder.int8_calibrator = None
            builder.strict_type_constraints = True
            # print(network.num_layers)
            for i in range(network.num_layers):
                layer = network[i]
                tensor = layer.get_output(0)
                if tensor:
                    tensor.set_dynamic_range(-1.0, 1.0)
                tensor = layer.get_input(0)
                if tensor:
                    tensor.set_dynamic_range(-1.0, 1.0)
            # print("pricision: int8")

        return builder.build_cuda_engine(network)


# Loads a test case into the provided pagelocked_buffer.
def load_normalized_test_case(test_image, pagelocked_buffer):
    # Converts the input image to a CHW Numpy array
    def normalize_image(image):
        # Resize, antialias and transpose the image to CHW.
        c, h, w = ModelData.INPUT_SHAPE
        image_arr = np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose(
            [2, 0, 1]).astype(trt.nptype(ModelData.DTYPE)).ravel()
        # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
        return (image_arr / 255.0 - 0.45) / 0.225

    # Normalize the image and copy to pagelocked memory
    np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
    return test_image


# get the engine from onnx_path or plan_path
# if plan file dose not exsits, create it
def get_resnet50_engine(onnx_path):
    plan_path = "resnet50-b{}".format(args.batch_size) + \
        ("q" if args.q else "") + ".engine"
    # else create the engine and save it
    if not os.path.isfile(plan_path):
        engine = build_engine_onnx(ModelData.MODEL_PATH)
        serialized_engine = engine.serialize()
        with open(plan_path, "wb") as f:
            f.write(engine.serialize())
    else:
        with open(plan_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def run(data_loader, engine):
    batch_time = AverageMeter()
    gpu = AverageMeter()
    gpu_mem = AverageMeter()
    # Allocate buffers and create a CUDA stream.
    h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
    # Contexts are used to perform inference.
    input = torch.rand((args.batch_size,) + ModelData.INPUT_SHAPE)
    with engine.create_execution_context() as context:
        end = time.time()
        for i in range(args.loop):
            np.copyto(h_input, input.reshape(-1))
            do_inference(context, h_input, d_input, h_output, d_output, stream)
            batch_time.update(time.time() - end)
            end = time.time()
            # https://pypi.org/project/py3nvml/
            util_rate = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            # print(util_rate.gpu, util_rate.memory)
            gpu.update(util_rate.gpu)
            mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            gpu_mem.update(mem_info.used >> 20)

            if i % args.print_freq == 0 and not args.csv:
                print('[{}/{}] batch time {batch_time.val:.3f} s (avg:{batch_time.avg:.3f})'.format(
                    i, args.loop, batch_time=batch_time))
    # print summary
    if args.csv:
        print("{}, {:.3f}, {:.3f}, {:.3f}, {}".format(
            args.batch_size,
            args.loop * args.batch_size / batch_time.sum,
            batch_time.avg,
            gpu.avg, gpu_mem.avg))
    else:
        print("batchsize: {} ".format(args.batch_size))
        print("throughput: {:.3f} img/sec".format(args.loop *
                                                  args.batch_size / batch_time.sum))
        print("Latency: {:.3f} sec".format(batch_time.avg))
        # see https://forums.fast.ai/t/show-gpu-utilization-metrics-inside-training-loop-without-subprocess-call/26594
        # show gpu utilization metrics inside trianing loop
        print("GPU util: {:.3f} %, GPU mem: {} MiB".format(
            gpu.avg, gpu_mem.avg))


def main():
    global args
    args = parser.parse_args()

   # Set the data path to the directory that contains the trained models and test images for inference.
    _, data_files = common.find_sample_data(description="Runs a ResNet50 network with a TensorRT inference engine.", subfolder="resnet50", find_files=[
        "binoculars.jpeg", "reflex_camera.jpeg", "tabby_tiger_cat.jpg", "class_labels.txt"])
    labels_file = data_files[3]
    labels = open(labels_file, 'r').read().split('\n')

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
    #     # num_workers=args.workers,
    #     num_workers=0,
    #     pin_memory=True)

    with get_resnet50_engine(ModelData.MODEL_PATH) as engine:
        # Allocate buffers and create a CUDA stream.
        h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:
            # Load a normalized test case into the host input page-locked buffer.
            run(0, engine)

    return
    # define loss function (criterion)
    # criterion = nn.CrossEntropyLoss().cuda()

    validate(data_loader, resnet50)
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
