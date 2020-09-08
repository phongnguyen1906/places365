# PlacesCNN to predict the scene category, attribute, and class activation map in a single pass
# by Bolei Zhou, sep 2, 2017
# updated, making it compatible to pytorch 1.x in a hacky way

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import numpy as np
import cv2
from PIL import Image
from multiprocessing import Pool
from dataclasses import dataclass
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

 # hacky way to deal with the Pytorch 1.0 update
def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

def load_labels():
    # prepare all the labels
    # scene category relevant
    file_name_category = 'categories_places365.txt'
    if not os.access(file_name_category, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    file_name_category_path = os.path.join(current_dir, file_name_category)
    with open(file_name_category_path) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = 'IO_places365.txt'
    if not os.access(file_name_IO, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
        os.system('wget ' + synset_url)
    file_name_IO_path = os.path.join(current_dir, file_name_IO)
    with open(file_name_IO_path) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = 'labels_sunattribute.txt'
    if not os.access(file_name_attribute, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
        os.system('wget ' + synset_url)
    file_name_attribute_path = os.path.join(current_dir, file_name_attribute)
    with open(file_name_attribute_path) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = 'W_sceneattribute_wideresnet18.npy'
    if not os.access(file_name_W, os.W_OK):
        synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
        os.system('wget ' + synset_url)
    file_name_W_path = os.path.join(current_dir, file_name_W)
    W_attribute = np.load(file_name_W_path)

    return classes, labels_IO, labels_attribute, W_attribute


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def returnTF():
# load the image transformer
    tf = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf


def load_model():
    # this model has a last conv feature map as 14x14

    model_file = 'wideresnet18_places365.pth.tar'
    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
        os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')

    import wideresnet
    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    
    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(model._modules.items()):
        module = recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)
    
    model.eval()

    # the following is deprecated, everything is migrated to python36

    ## if you encounter the UnicodeDecodeError when use python3 to load the model, add the following line will fix it. Thanks to @soravux
    #from functools import partial
    #import pickle
    #pickle.load = partial(pickle.load, encoding="latin1")
    #pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    #model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)

    return model

# load the labels
classes, labels_IO, labels_attribute, W_attribute = load_labels()

# load the transformer
tf = returnTF() # image transformer

@dataclass
class ExtractPlaceCNNFeatureParams:
    image_file_path: str
    raw_feat_output_path: str
    output_attribute_feat: bool = False
    attribute_feat_output_path: str = None
    attribute_pred_output_path: str = None
    output_category_feat: bool = False
    category_feat_output_path: str = None
    category_pred_output_path: str = None
    output_CAMs: bool = False
    CAMs_output_path: str = None



def extract_placeCNN_feature(params: ExtractPlaceCNNFeatureParams, model):
    if os.path.exists(params.raw_feat_output_path):
        return

    features_blobs = []
    # hook the feature extractor
    def hook_feature(module, input, output):
        features_blobs.append(np.squeeze(output.data.cpu().numpy()))

    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)

    # Load images
    try:
        img = Image.open(params.image_file_path)
    except Exception as e:
        with open('logs.txt', 'a') as f:
            print(e, file=f)
        return
    
    # Handle some special image format in PIL Image library and convert them into RGB to process properly
    img = img.convert('RGB')
    input_img = V(tf(img).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()

    # Raw feature after processing through average pooling layer before transmitting to fully-connected layers for classification
    raw_feature = features_blobs[1]
    np.save(params.raw_feat_output_path, raw_feature)

    # Scene category vector feature
    if params.output_category_feat:
        response_category = h_x.cpu().numpy()
        np.save(params.category_feat_output_path, response_category)
        # Output predictions of scene categories
        probs, idx = h_x.sort(0, True)
        probs = probs.numpy()
        idx = idx.numpy()
        with open(params.category_pred_output_path, 'w') as f:
            n_preds = 5
            for index in range(n_preds):
                print('{:.3f} -> {}'.format(probs[index], classes[idx[index]]), file=f)
        # CAMs heatmap output
        if params.output_CAMs:
            # get the softmax weight
            _params = list(model.parameters())
            weight_softmax = _params[-2].data.numpy()
            weight_softmax[weight_softmax<0] = 0
            CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
            _img = cv2.imread(params.image_file_path)
            height, width, _ = _img.shape
            heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
            result = heatmap * 0.4 + _img * 0.5
            cv2.imwrite(params.CAMs_output_path, result)

    # Scence attribute vector feature
    if params.output_attribute_feat:
        responses_attribute = W_attribute.dot(raw_feature)
        np.save(params.attribute_feat_output_path, responses_attribute)
        # Output predictions of scene attributes
        idx_a = np.argsort(responses_attribute)
        with open(params.attribute_pred_output_path, 'w') as f:
            n_preds = 10
            print(', '.join([labels_attribute[idx_a[index]] for index in range(-1, -n_preds, -1)]), file=f)



if __name__ == '__main__':

    # load the model
    model = load_model()

    # load the test image
    # img_url = 'http://places.csail.mit.edu/demo/6.jpg'
    # os.system('wget %s -q -O test.jpg' % img_url)

    #change into working directory
    os.chdir('G:/Data_Science_Project/CenterNet_Paper')

    attribute_output_folder_path = 'Attributes'
    attribute_feat_output_folder = os.path.join(attribute_output_folder_path, 'feat')
    if not os.path.exists(attribute_feat_output_folder):
        os.makedirs(attribute_feat_output_folder)
    attribute_pred_output_folder = os.path.join(attribute_output_folder_path, 'pred')
    if not os.path.exists(attribute_pred_output_folder):
        os.makedirs(attribute_pred_output_folder)

    category_output_folder_path = 'Categories'
    category_feat_output_folder = os.path.join(category_output_folder_path, 'feat')
    if not os.path.exists(category_feat_output_folder):
        os.makedirs(category_feat_output_folder)
    category_pred_output_folder = os.path.join(category_output_folder_path, 'pred')
    if not os.path.exists(category_pred_output_folder):
        os.makedirs(category_pred_output_folder)

    raw_output_folder_path = 'Raw'
    if not os.path.exists(raw_output_folder_path):
        os.makedirs(raw_output_folder_path)
    
    CAMs_output_path = 'CAMs'
    if not os.path.exists(CAMs_output_path):
        os.makedirs(CAMs_output_path)

    img_dir = 'G:/Data_Science_Project/CenterNet_Paper/images' #image_folder
    for root, dirs, files in os.walk(img_dir):

        for dir in dirs:
            # print(dir)
            image_raw_output_path = os.path.join(raw_output_folder_path, dir)
            if not os.path.exists(image_raw_output_path):
                os.makedirs(image_raw_output_path)

            image_attribute_feat_path = os.path.join(attribute_feat_output_folder, dir)
            if not os.path.exists(image_attribute_feat_path):
                os.makedirs(image_attribute_feat_path)
            image_attribute_pred_path = os.path.join(attribute_pred_output_folder, dir)
            if not os.path.exists(image_attribute_pred_path):
                os.makedirs(image_attribute_pred_path)

            image_category_feat_path = os.path.join(category_feat_output_folder, dir)
            if not os.path.exists(image_category_feat_path):
                os.makedirs(image_category_feat_path)
            image_category_pred_path = os.path.join(category_pred_output_folder, dir)
            if not os.path.exists(image_category_pred_path):
                os.makedirs(image_category_pred_path)

            image_CAMs_path = os.path.join(CAMs_output_path, dir)
            if not os.path.exists(image_CAMs_path):
                os.makedirs(image_CAMs_path)

            for file in os.listdir(os.path.join(root,dir)):
                # print(file)
                image_path = os.path.join(root, dir, file)
                image_name = Path(image_path).resolve().stem
                params = ExtractPlaceCNNFeatureParams(
                    image_file_path=image_path,
                    raw_feat_output_path=os.path.join(image_raw_output_path, f'{image_name}.npy'),
                    output_attribute_feat=True,
                    attribute_feat_output_path=os.path.join(image_attribute_feat_path, f'{image_name}.npy'),
                    attribute_pred_output_path=os.path.join(image_attribute_pred_path, f'{image_name}.txt'),
                    output_category_feat=True,
                    category_feat_output_path=os.path.join(image_category_feat_path,  f'{image_name}.npy'),
                    category_pred_output_path=os.path.join(image_category_pred_path, f'{image_name}.txt'),
                    output_CAMs=True,
                    CAMs_output_path=os.path.join(image_CAMs_path, f'{image_name}.jpg')
                )
                extract_placeCNN_feature(params, model)
                print(f'extracted {image_path}')
