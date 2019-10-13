# ref: https://pytorch.org/docs/stable/onnx.html

import torchvision.models as models
import onnx

import torch
import torchvision

image_size = 224


def main():
    dummpy_input = torch.randn(
        1, 3, image_size, image_size, device='cuda')
    # resnet:
    #  output shape: (1000)
    resnet50 = models.resnet50(pretrained=True).cuda()
    # print(resnet50)

    # The resulting alexnet.onnx is a binary protobuf file which contains both
    # the network structure and parameters of the model you exported
    torch.onnx.export(resnet50, dummpy_input, "resnet50.onnx",
                      verbose=True)

    onnx_model = onnx.load("resnet50.onnx")
    # onnx.checker.check_model(onnx_model)    # segfault, pytorch1.2 pytorch1.3
    # onnx.helper.printable_graph(model.graph)


def check_model(model_path):
    model = onnx.load(model_path)
    # onnx.checker.check_model(model)
    # onnx.helper.printable_graph(model.graph)


if __name__ == '__main__':
    # main()
    check_model("ResNet50.onnx")
