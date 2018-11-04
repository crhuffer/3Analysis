import torch
import torch.nn as nn
import numpy as np
import re
import cv2
from torch.autograd import Variable


class empty_layer(nn.Module):
    def __init__(self):
        super().__init__()


class yolo_layer(nn.Module):
    def __init__(self, anchors):
        super().__init__()
        self.anchors = anchors


def load_classes(namesfile):
    with open(namesfile, "r") as file:
        names = file.read().split("\n")[:-1]
    return names

# pass config file into a list
def parse_cfg(cfgfile):
    with open(cfgfile, "r") as file:
        lines = file.read().split('\n')
        # get rid of lines that are blanks for comments
        lines = [line for line in lines if (len(line) > 0) and
                 (line[0] != '#')]

        # get ride of white spaces from the back and front
        lines = [line.lstrip().rstrip() for line in lines]
        temp_block = {}
        blocks = []

        # loop through the lines to fill up the blocks list
        for line in lines:
            if line[0] == "[":
                if len(temp_block) != 0:
                    blocks.append(temp_block)
                    temp_block = {}
                temp_block['type'] = line[1:-1].lstrip().rstrip()
            else:
                key, value = line.split('=')

                # set key value to the dictionary, remove possible spaces
                # near '='
                temp_block[key.rstrip()] = value.lstrip()
        blocks.append(temp_block)  # append the final block when the loop is over
        return blocks


def conv_layer_handling(module, index, layer, in_channel, layer_type_dic):
    activation = layer["activation"]
    try:
        batch_norm = layer["batch_normalize"]
        bias = False  # bias term is already included in batch_norm
                      # gamma * normalized(x) + bias
    except KeyError as e:
        batch_norm = 0
        bias = True

    out_channels = int(layer["filters"])
    kernel_size = int(layer["size"])
    padding = 0 if kernel_size == 1 else 1  # padding not require for 1*1 conv
    stride = int(layer["stride"])
    conv = nn.Conv2d(in_channel, out_channels, kernel_size, stride, padding,
                     bias=bias)
    module.add_module("conv_{}".format(index), conv)

    # batch norm layer if applicable
    if batch_norm:
        bn = nn.BatchNorm2d(out_channels)
        module.add_module("bach_norm_{}".format(index), bn)

    if activation == "leaky":
        activn = nn.LeakyReLU(0.1)
        module.add_module("leaky_{}".format(index), activn)
    layer_type_dic["conv"].append(index)
    return out_channels


def upsample_layer_handling(module, index, layer, layer_type_dic):
    stride = int(layer["stride"])
    upsample = nn.Upsample(scale_factor=stride, mode="nearest")
    module.add_module("upsample_{}".format(index), upsample)
    layer_type_dic['upsampling'].append(index)


def route_layer_handling(module, index, layer, out_channels_list,
                         layer_type_dic):
    items = [int(item) for item in (layer["layers"].split(','))]
    first_layer = index + items[0]
    out_channel = out_channels_list[first_layer]
    if len(items) == 1:
        layer_type_dic['referred_relationship'][index] = (first_layer)
        layer_type_dic['route_1'].append(index)
    elif len(items) == 2:
        out_channel += out_channels_list[items[1]]  # sum of c from two layers
        layer_type_dic['referred'].append(items[1])
        layer_type_dic['referred_relationship'][index] = \
            ((first_layer, items[1]))   # [items[1] is the second layer
        layer_type_dic['route_2'].append(index)
    else:
        raise Exception('Route layer is not behaving as we planned, please \
                        change the code according.')
    route = empty_layer()
    module.add_module("route_{}".format(index), route)
    layer_type_dic['referred'].append(index + items[0])
    return out_channel


def shortcut_layer_handling(module, index, layer, layer_type_dic):
    short_cut = empty_layer()
    from_layer = int(layer["from"])
    module.add_module("short_cut_{}".format(index), short_cut)
    layer_type_dic['shortcut'].append(index)
    # add the layer that is referred in the short cut layer
    layer_type_dic['referred'].append(index + from_layer)
    layer_type_dic['referred_relationship'][index] = (index + from_layer)


def yolo_layer_handling(module, index, layer, layer_type_dic):
    anchor_index = [int(x) for x in (layer["mask"].split(","))]
    anchors = re.split(',  |,', layer["anchors"])
    anchors = np.reshape([int(x) for x in anchors], (9, 2))
    anchors = anchors[anchor_index]
    num_anchors = len(anchors)
    classes = int(layer["classes"])
    layer_type_dic["net_info"]["num_classes"] = classes
    layer_type_dic["net_info"]["num_anchors"] = num_anchors
    yolo = yolo_layer(anchors)
    module.add_module("yolo_{}".format(index), yolo)
    layer_type_dic['yolo'].append(index)


def create_module(blocks):
    module_list = nn.ModuleList()
    layer_type_dic = {"net_info": {},
                      "conv": [],
                      "upsampling": [],
                      "shortcut": [],
                      "route_1": [],
                      "route_2": [],
                      "yolo": [],
                      "referred": [],
                      "referred_relationship": {}}
    layer_type_dic["net_info"] = blocks[0]
    layer_type_dic["net_info"]["height"] = int(
            layer_type_dic["net_info"]["height"]
            )
    in_channel = 3
    out_channels_list = []

    for index, layer in enumerate(blocks[1:]):
        module = nn.Sequential()

        # conv layer
        if layer["type"] == "convolutional":
            in_channel = conv_layer_handling(module, index, layer, in_channel,
                                             layer_type_dic)

        # up sampling layer
        elif layer["type"] == "upsample":
            upsample_layer_handling(module, index, layer, layer_type_dic)

        # route layer
        elif layer["type"] == "route":
            in_channel = route_layer_handling(module, index, layer,
                                              out_channels_list,
                                              layer_type_dic)

        # shor cut layer
        elif layer["type"] == "shortcut":
            shortcut_layer_handling(module, index, layer, layer_type_dic)

        # yolo layer
        elif layer["type"] == "yolo":
            yolo_layer_handling(module, index, layer, layer_type_dic)

        out_channels_list.append(in_channel)
        module_list.append(module)

    for index, value in layer_type_dic.items():
        if index not in ["net_info", "referred_relationship"]:
            layer_type_dic[index] = list(layer_type_dic[index])

    return layer_type_dic, module_list


def box_iou(box1, box2):
    ''' Both boxes need to be a 2d tensor '''
    b1x_min, b1y_min, b1x_max, b1y_max = box1[:, 0], box1[:, 1],\
        box1[:, 2], box1[:, 3]
    b2x_min, b2y_min, b2x_max, b2y_max = box2[:, 0], box2[:, 1], \
        box2[:, 2], box2[:, 3]

    # find the co-ordinates of the intersection rectangle
    inter_box_xmin = torch.max(b1x_min, b2x_min)
    inter_box_ymin = torch.max(b1y_min, b2y_min)
    inter_box_xmax = torch.min(b1x_max, b2x_max)
    inter_box_ymax = torch.min(b1y_max, b2y_max)

    # intersection area
    inter_area = torch.clamp((inter_box_xmax - inter_box_xmin + 1),
                             min=0) * \
        torch.clamp((inter_box_ymax - inter_box_ymin + 1), min=0)

    # intersection area
    box1_area = (b1x_max - b1x_min + 1) * (b1y_max - b1y_min + 1)
    box2_area = (b2x_max - b2x_min + 1) * (b2y_max - b2y_min + 1)
    union_area = box1_area + box2_area - inter_area

    # iou
    iou = inter_area / union_area
    return iou
#
#
#def filter_results(prediction, obj_threshhold, num_classes,  nms_threshold):
#    prediction[prediction[:, :, 4] < obj_threshhold] = 0
#    # create a new tensor that has the identical dtype as prediction tensor
#    box_corners = prediction.new(prediction.shape)
#    batch_size = box_corners.size(0)
#
#    # convert tensor format from center x, center y to xmin,ymin
#    box_corners[:, :, :2] = prediction[:, :, :2] - prediction[:, :, 2:4]/2
#
#    # convert tensor format from width. height xmax,ymax
#    box_corners[:, :, 2:4] = prediction[:, :, :2] + prediction[:, :, 2:4]/2
#    # copy the transfored numbers to prediction
#    prediction[:, :, :4] = box_corners[:, :, :4]
#    write = False
#    for i in range(batch_size):
#        image_pred = prediction[i]
#        image_pred = image_pred[image_pred[:, 4] != 0]
#        if image_pred.shape[0] == 0:
#            continue
#        class_prob, class_index = torch.max(image_pred[:, 5:5+num_classes], 1)
#        class_prob = class_prob.float().view(-1, 1)
#        class_index = class_index.float().view(-1, 1)
#        # concat xmin,ymin,xmax,ymax,bc together with class prob and class index
#        image_pred = torch.cat(
#                              (image_pred[:, :5], class_prob, class_index),
#                              1)
#        # find the unique index in the picture thus find the classes
#        img_classes = torch.unique(image_pred[:, 6])
#        for cls in img_classes:
#            # find the predidction rows that belongs to the same class
#            image_pred_class = image_pred[image_pred[:, 6] == cls]
#            # predidction row index sorted by its object confidence
#            conf_sorted_index = torch.sort(
#                                           image_pred_class[:, 4],
#                                           descending=True
#                                           )[1]
#            num_of_dections = len(conf_sorted_index)
#            image_pred_class = image_pred_class[conf_sorted_index]
#            for j in range(1, num_of_dections+1):
#                try:
#                    iou = box_iou(image_pred_class[j-1, :4].view(1, -1),
#                                  image_pred_class[j:, :4])
#                    image_pred_class[j:, :][iou > nms_threshold] = 0
#                    image_pred_class = image_pred_class[image_pred_class[:, 4] != 0]
#                except (IndexError, ValueError):
#                    break
#            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(i)
#            seq = batch_ind, image_pred_class
#            if not write:
#                output = torch.cat(seq, 1)
#                write = True
#            else:
#                out = torch.cat(seq, 1)
#                output = torch.cat((output, out))
#    try:
#        return output
#    except NameError:
#        return 0
#


def filter_results(prediction, confidence, num_classes, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask

    # create a new tensor that has the identical dtype as prediction tensor
    box_corners = prediction.new(prediction.shape)

    # convert tensor format from center x, center y to xmin,ymin
    box_corners[:, :, :2] = prediction[:, :, :2] - prediction[:, :, 2:4]/2

    # convert tensor format from width. height xmax,ymax
    box_corners[:, :, 2:4] = prediction[:, :, :2] + prediction[:, :, 2:4]/2

    # copy the transfored numbers to prediction
    prediction[:,:,:4] = box_corners[:,:,:4]

    batch_size = prediction.size(0)

    write = False



    for ind in range(batch_size):
        # image Tensor
        image_pred = prediction[ind]

        # ge
        max_conf, class_index = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        class_index = class_index.float().unsqueeze(1)

        # concat xmin,ymin,xmax,ymax,bc together with class prob and class index
        seq = (image_pred[:,:5], max_conf, class_index)
        image_pred = torch.cat(seq, 1)

        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue
#

#        # find the unique index in the picture thus find the classes
        img_classes = torch.unique(image_pred_[:, 6])


        for cls in img_classes:
            #perform NMS


            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)

            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #Number of detections

            for i in range(idx):
                # Get the IOUs of all boxes that come after the one we are
                # looking at in the loop
                try:
                    ious = box_iou(image_pred_class[i, :4].view(1, -1),
                                   image_pred_class[i+1:, :4])

                except (IndexError, ValueError):
                    break

## FIXME: I belive this comment has the equality backwards.
                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask

                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
            # Repeat the batch_id for the same class in the image
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))

    try:
        return output
    except NameError:
        return 0


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image

    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (608,608))          # Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       # Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     # Convert to float                   # Convert to Variable
    return img_

