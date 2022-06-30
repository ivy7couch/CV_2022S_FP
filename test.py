from __future__ import division
import argparse
from matplotlib.pyplot import annotate
from sqlalchemy import true
import torch
import os
import cv2
import numpy as np
# import dlib
from models.basenet import BaseNet, MobileNet
import pickle
import torchvision.models as models

parser = argparse.ArgumentParser(description='PyTorch face landmark')

parser.add_argument('-c', '--checkpoint', default='checkpoint/facesynthetics/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
# parser.add_argument('-c', '--checkpoint', default='checkpoint/facesynthetics/epoch0.pth', type=str, metavar='PATH',
#                     help='path to save checkpoint (default: checkpoint)')
parser.add_argument('-img', '--image', default='facesynthetics', type=str)
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpu_id', default='0,1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')


args = parser.parse_args()
mean = np.asarray([0.4465, 0.4822, 0.4914])
std = np.asarray([0.1994, 0.1994, 0.2023])


def load_model():
    model = MobileNet()
    # model = models.mobilenet_v2(pretrained=True) 
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    # checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint['state_dict'])
    # model = torch.load(args.checkpoint)
    # pretrained_dict = model_old.state_dict()
    # pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
    # model.load_state_dict(model)
    return model

def predict_68(test_folder):
    draw_point = True
    predict_pts = []
    for idx, filename in enumerate(os.listdir(test_folder)):
        img = cv2.imread(os.path.join(test_folder,filename))
        raw_img = img
        img = img/255.0
        img = (img-mean)/std
        img = img.transpose((2, 0, 1))
        img = img.reshape((1,) + img.shape)
        input = torch.from_numpy(img).float()
        input= torch.autograd.Variable(input)
        out = model(input).cpu().data.numpy()
        out = out.reshape(-1,2)
        
        if idx == 0:
            print("output = ", out*384)  # check output
        raw_img = cv2.resize(raw_img,(out_size,out_size))
        
        if draw_point:
            for i in range(68):
                cv2.circle(raw_img,(int(out[i][0]*out_size),int(out[i][1]*out_size)),1,(255,0,0),-1)
            cv2.imwrite(f'./result_1/result_{filename}.png', raw_img)
        
        out = (out.reshape(136,1).squeeze(axis=1))*384
        out = [str(i) for i in out]
        predict_pts.append([filename])
        predict_pts[idx].extend(out)
    
    # write to txt
    with open('output.txt', 'w') as f:
        f.write('\n'.join([' '.join(i) for i in predict_pts]))
        
        
def calculate_nme(val_folder):
    # annot should be outside val_folder
    with open('../pytorch_face_landmark/data/annot.pkl', 'rb') as f:
        annot = pickle.load(f)
        X, Y = annot
        # print(X)
    total_nme = 0
    for idx, filename in enumerate(X):
        gt = np.array([list(x) for x in Y[idx]])
        img = cv2.imread(os.path.join(val_folder,filename))
        # raw_img = img
        img = img/255.0
        img = (img-mean)/std
        img = img.transpose((2, 0, 1))
        img = img.reshape((1,) + img.shape)
        input = torch.from_numpy(img).float()
        input= torch.autograd.Variable(input)
        out = model(input).cpu().data.numpy()
        out = out.reshape(-1,2)
        pred = out*384
        
        # calculate nme per image
        dis = (gt - pred)
        # print("dis = ", np.power(dis, 2))
        # xxx
        dis = np.sqrt(np.sum(np.power(dis, 2), 1))
        dis = np.mean(dis)
        nme = dis / 384
        total_nme += nme
    
    print("total_nme = ", total_nme)
    
        

if __name__ == '__main__':
    out_size = 256
    model = load_model()
    model = model.eval()
    
    # calculate validation nme
    val_path = '../pytorch_face_landmark/data/aflw_val2/'
    calculate_nme(val_path)
    xxx    
    
    # predict 68 facial landmaarks
    test_path = '../pytorch_face_landmark/data/aflw_test/'  # path to testing data
    predict_68(test_path)
    
    # check output file
    cnt = 0
    with open('output.txt') as f:
        for line in f.readlines():
            s = line.split(' ')
            print(s[1], s[2])
            xxx
            cnt += 1
    print(cnt)
    
