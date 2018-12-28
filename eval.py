import os

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import click
import numpy as np
from pdb import set_trace as st
import h5py
import utils
from tensorboardX import SummaryWriter
import argparse
from pspnet import PSPNet

models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet', inchannel = 1),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet', inchannel = 1),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18', inchannel = 1),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34', inchannel = 1),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50', inchannel = 1),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101', inchannel = 1),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152', inchannel = 1)
}


def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        print("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda()
    return net, epoch

def evaluate(model_path, backend, batch_size, data_path, resize):
    net, starting_epoch = build_network(model_path, backend)
    test_iterator = utils.load_data(os.path.join(data_path, 'test.h5'), batch_size, resize, shuffle=False)
    
    OA_all_1 = []
    OA_all_2 = []
    meanIU_all_1 = []
    meanIU_all_2 = []
    writer = SummaryWriter()
    count = 0
    for batch_idx, (x, y, y_cls) in enumerate(test_iterator):
        x, y, y_cls = Variable(x).cuda(), Variable(y).cuda(), Variable(y_cls).cuda()
        out, _ = net(x)
        # convert to cpu
        y = y.cpu().numpy()
        out_map = np.argmax(out.detach().cpu().numpy(), axis = 1)
        for gt, pred in zip(y, out_map):
            OA = utils.comp_OA(gt, pred)
            meanIU = utils.comp_meanIU(gt, pred)
            OA_all_1.append(OA[0])
            meanIU_all_1.append(meanIU[0])
            try:
                OA_all_2.append(OA[1])
                meanIU_all_2.append(meanIU[1])
            except:
                pass
        for i, (imx, imy, imp) in enumerate(zip(x, y, out_map) ):
            imx = imx.cpu().numpy()
            writer.add_image(str(count+i)+'_Input', utils.resize_singleim(utils.unnorm_im(imx), 425, 904 ), 0)
            writer.add_image(str(count+i)+'_Output', utils.resize_singleim((imp*0.5).astype(float), 425, 904 ), 0)
            writer.add_image(str(count+i)+'_GT', utils.resize_singleim((imy*0.5).astype(float), 425, 904 ), 0)
        count += batch_size
    print('Val Eyelid Overall Accuracy: '+str(100* np.mean(OA_all_1)))
    print('Val Atrophy Overall Accuracy: '+str(100* np.mean(OA_all_2)))
    print('Val Eyelid MeanIU: '+str(100* np.mean(meanIU_all_1)))
    print('Val Atrophy MeanIU: '+str(100* np.mean(meanIU_all_2)))
        
    writer.add_scalar('data/OA_all_1', np.mean(OA_all_1), 0)
    writer.add_scalar('data/OA_all_2', np.mean(OA_all_2), 0)
    writer.add_scalar('data/meanIU_all_1', np.mean(meanIU_all_1), 0)
    writer.add_scalar('data/meanIU_all_2', np.mean(meanIU_all_2), 0)
        
    writer.close()

if __name__ == '__main__':    
    #eval settings
    parser = argparse.ArgumentParser(description='Meibography Eval')
    parser.add_argument('--batch_size', default=5 , type=int, metavar='N', help='Batch size of test set')
    parser.add_argument('--resize', default=400 , type=int, metavar='N', help='Resize image to what dimension')    
    parser.add_argument('--model_path', type=str, metavar='XXX', help='Path to the model')
    parser.add_argument('--backend', default='resnet34', type=str, metavar='XXX', help='Backend architecture')
    parser.add_argument('--data_path', default='/home/peterwg/dataset/meibo2018', type=str, metavar='XXX', help='Path to dataset folder')
    args = parser.parse_args()
    evaluate(args.model_path, args.backend, args.batch_size, args.data_path, args.resize)