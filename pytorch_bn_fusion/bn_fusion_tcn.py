import torch
import torch.nn as nn


def fuse_bn_sequential(model):
    """
    This function takes a sequential block and fuses the batch normalization with convolution

    :param model: nn.Sequential. Source resnet model
    :return: nn.Sequential. Converted block
    """
    for m in model.children():
        for c in m.children():
            for i in range(len(c)):
                # First depthwise seperable convolution within the block
                conv11, bn11 = c[i].DSConv1[0], c[i].DSConv1[1]
                conv12, bn12 = c[i].DSConv1[3], c[i].DSConv1[4]

                # Second depthwise seperable convolution within the block
                conv21, bn21 = c[i].DSConv2[0], c[i].DSConv2[1]
                conv22, bn22 = c[i].DSConv2[3], c[i].DSConv2[4]

                fused_conv11 = torch.nn.utils.fuse_conv_bn_eval(conv11, bn11)
                fused_conv12 = torch.nn.utils.fuse_conv_bn_eval(conv12, bn12)

                fused_conv21 = torch.nn.utils.fuse_conv_bn_eval(conv21, bn21)
                fused_conv22 = torch.nn.utils.fuse_conv_bn_eval(conv22, bn22)

                
                # fused_conv11.weight = torch.nn.Parameter((torch.diagonal(fused_conv11.weight)).transpose(0,2).transpose(1,2))
                # fused_conv12.weight = torch.nn.Parameter((torch.diagonal(fused_conv12.weight)).transpose(0,2).transpose(1,2))
                # fused_conv21.weight = torch.nn.Parameter((torch.diagonal(fused_conv21.weight)).transpose(0,2).transpose(1,2))
                # fused_conv22.weight = torch.nn.Parameter((torch.diagonal(fused_conv22.weight)).transpose(0,2).transpose(1,2))
                # fused_conv11.weight = torch.nn.Parameter(torch.diagonal(fused_conv11.weight))
                # fused_conv12.weight = torch.nn.Parameter(fused_conv12.weight.squeeze(0))
                # fused_conv21.weight = torch.nn.Parameter(fused_conv21.weight.squeeze(0))
                # fused_conv22.weight = torch.nn.Parameter(fused_conv22.weight.squeeze(0))

                c[i].DSConv1[0] = fused_conv11
                c[i].DSConv1[1] = nn.Identity()
                c[i].DSConv1[3] = fused_conv12
                c[i].DSConv1[4] = nn.Identity()

                c[i].DSConv2[0] = fused_conv21
                c[i].DSConv2[1] = nn.Identity()
                c[i].DSConv2[3] = fused_conv22
                c[i].DSConv2[4] = nn.Identity()



    return model

# def fuse_bn_recursively(model):
#     for module_name in model._modules:
#         model._modules[module_name] = fuse_bn_sequential(model._modules[module_name], model)
#         if len(model._modules[module_name]._modules) > 0:
#             fuse_bn_recursively(model._modules[module_name])

#     return model