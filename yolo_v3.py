from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from utilis import parse_cfg, create_module, get_test_input
from torch.autograd import Variable
import numpy as np

# We are inhereting the properties from torch.nn.Module
class yolo_v3(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks
        self.layer_type_dic, self.module_list = create_module(blocks)
        self.num_anchors = self.layer_type_dic["net_info"]["num_anchors"]
        self.num_classes = self.layer_type_dic["net_info"]["num_classes"]

    def predict(self, x, index, layer, batch_size, input_size, cuda):
        anchors = layer[0].anchors
        yolo_size = x.size(2)
        stride = input_size // yolo_size
        depth = 5 + self.num_classes
        wh = yolo_size**2
        x = x.view(batch_size, depth*self.num_anchors, wh)
        x = x.transpose(1, 2).contiguous().view(
                                              batch_size,
                                              wh*self.num_anchors,
                                              depth
                                              )
        # x y centre point must be within 0 to 1, same to the object confidence
        x[:, :, 0] = torch.sigmoid(x[:, :, 0])  # centre x
        x[:, :, 1] = torch.sigmoid(x[:, :, 1])  # centre y
        x[:, :, 4] = torch.sigmoid(x[:, :, 4])  # object confidence

        # offset the centre coordinates according to their grid
# =============================================================================
#         offset = torch.FloatTensor(np.arange(yolo_size)).unsqueeze(1)
#         offset = offset.repeat(yolo_size, 2*self.num_anchors).view(1, -1, 2)
#         if cuda:
#             offset = offset.cuda()
#         x[:, :, :2] += offset
# =============================================================================
        grid = np.arange(yolo_size)
        a,b = np.meshgrid(grid, grid)

        x_offset = torch.FloatTensor(a).view(-1,1)
        y_offset = torch.FloatTensor(b).view(-1,1)

        if cuda:
            x_offset = x_offset.cuda()
            y_offset = y_offset.cuda()

        x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,self.num_anchors).view(-1,2).unsqueeze(0)

        x[:,:,:2] += x_y_offset

        # treansform the width and height with anchors (bw = pw * e^tw)
        anchors = [anchor/stride for anchor in anchors]
        anchors = torch.FloatTensor(anchors)   # in order to use repeat
        if cuda:
            anchors = anchors.cuda()
        anchors = anchors.repeat(wh, 1).unsqueeze(0)
        x[:, :, 2:4] = torch.exp(x[:, :, 2:4])*anchors

        # sigmoid class confidence
        x[:, :, 5:] = torch.sigmoid(x[:, :, 5:])

        # standadize to the imput size
        x[:, :, :4] *= stride
        return x


    def forward(self, x, cuda):
        cache = {}
        input_size = self.layer_type_dic["net_info"]["height"]
        batch_size = x.size(0)
        for index, layer in enumerate(self.module_list):
            if index in self.layer_type_dic["conv"] or \
               index in self.layer_type_dic["upsampling"]:
                    x = self.module_list[index](x)

            elif index in self.layer_type_dic["route_1"]:
                referred_layer = self.layer_type_dic[
                                                    "referred_relationship"
                                                    ][index]
                x = cache[referred_layer]
            elif index in self.layer_type_dic["route_2"]:
                referred_layer = self.layer_type_dic[
                                                     "referred_relationship"
                                                     ][index]
                x = torch.cat((cache[referred_layer[0]],
                               cache[referred_layer[1]]), 1)

            elif index in self.layer_type_dic["shortcut"]:
                referred_layer = self.layer_type_dic[
                                                    "referred_relationship"
                                                    ][index]
                x += cache[referred_layer]

            elif index in self.layer_type_dic["yolo"]:
                x = self.predict(x, index, layer, batch_size, input_size, cuda)
                if index == self.layer_type_dic["yolo"][0]:
                    detections = x
                else:
                    detections = torch.cat((detections, x), 1)


            if index in self.layer_type_dic["referred"]:
                cache[index] = x
        return detections

    def load_weights(self, weightfile):
        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        weights = np.fromfile(fp, dtype=np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            #If module_type is convolutional load weights
            #Otherwise ignore.

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]


                if (batch_normalize):
                    bn = model[1]

                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    #Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    #Number of biases
                    num_biases = conv.bias.numel()

                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


cfg_path = "../4Others/yolov3.cfg"
blocks = parse_cfg(cfg_path)







