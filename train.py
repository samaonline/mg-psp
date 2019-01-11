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

from pspnet import PSPNet

sampler_dic = {'type': 'ClassAwareSampler', 'def_file': './data/ClassAwareSampler.py', 'num_samples_cls': 2}

models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet', inchannel = 1),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet', inchannel = 1),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18', inchannel = 1),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, all_feat_size=771, backend='resnet34', inchannel = 1),
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
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda()
    return net, epoch

@click.command()
@click.option('--data_path', type=str, default='/home/peterwg/dataset/meibo2018', help='Path to dataset folder')
@click.option('--models_path', type=str, default='./resnet34', help='Path for storing model snapshots')
@click.option('--backend', type=str, default='resnet34', help='Feature extractor')
@click.option('--snapshot', type=str, default=None, help='Path to pretrained weights')
@click.option('--resize', type=int, default=420, help='Resize the original image to certain size')
# add random crop and center crop to see if acc improves
@click.option('--crop', type=int, default=400, help='Random (train) / center (val) crop size')
@click.option('--threshold', type=float, default=0.1, help='Treshold for classifying meiboscore 0')
@click.option('--batch_size', type=int, default=10)
@click.option('--batch_size_test', type=int, default=5)
@click.option('--alpha', type=float, default=0.4, help='Coefficient for classification loss term')
@click.option('--beta', type=float, default=0.4, help='Coefficient for mg classification loss term')
@click.option('--epochs', type=int, default=100, help='Number of training epochs to run')
@click.option('--gpu', type=str, default='0,5', help='List of GPUs for parallel training, e.g. 0,1,2,3')
@click.option('--start_lr', type=float, default=0.001)
@click.option('--milestones', type=str, default='50,75,93', help='Milestones for LR decreasing')
#@click.option('--name', type=str, default=None, help='Name of the exp')
@click.option('--log_interval', type=int, default='20', help='Interval of batches to print log')
def train(data_path, models_path, backend, snapshot, resize, batch_size, batch_size_test, alpha, beta, epochs, start_lr, milestones, gpu,log_interval,  crop, threshold):
    #os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    net, starting_epoch = build_network(snapshot, backend)
    data_path = os.path.abspath(os.path.expanduser(data_path))
    models_path = os.path.abspath(os.path.expanduser(models_path))
    os.makedirs(models_path, exist_ok=True)
    
    '''
        To follow this training routine you need a DataLoader that yields the tuples of the following format:
        (Bx3xHxW FloatTensor x, BxHxW LongTensor y, BxN LongTensor y_cls) where
        x - batch of input images,
        y - batch of groung truth seg maps,
        y_cls - batch of 1D tensors of dimensionality N: N total number of classes, 
        y_cls[i, T] = 1 if class T is present in image i, 0 otherwise
    '''
    #train_loader, class_weights, n_images = None, None, None
    class_weights = None
    train_iterator = utils.load_data(os.path.join(data_path, 'train.h5'), batch_size, resize, sampler_dic)
    val_iterator, names = utils.load_data_nnames(os.path.join(data_path, 'test.h5'), batch_size_test, resize, shuffle=False)
    
    optimizer = optim.Adam(net.parameters(), lr=start_lr)
    scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in milestones.split(',')])
    writer = SummaryWriter(models_path)
    for epoch in range(starting_epoch, starting_epoch + epochs):
        seg_criterion = nn.NLLLoss(weight=class_weights)
        cls_criterion = nn.BCEWithLogitsLoss(weight=class_weights)
        meibocls_criterion = nn.CrossEntropyLoss()
        epoch_losses = []
        #train_iterator = tqdm(loader, total=max_steps // batch_size + 1)
        net.train()
        for batch_idx, (x, y, y_cls, ms) in enumerate(train_iterator):
            #steps += batch_size
            optimizer.zero_grad()
            x, y = utils.random_crop(x, y, crop)
            x, y, y_cls, ms = Variable(x).cuda(), Variable(y).cuda(), Variable(y_cls).cuda(), Variable(ms).cuda()
            out, out_cls, feat = net(x)
            seg_loss, cls_loss, mscls_loss = seg_criterion(out, y), cls_criterion(out_cls, y_cls), meibocls_criterion(feat, ms)
            loss = seg_loss + alpha * cls_loss + beta * mscls_loss
            epoch_losses.append(loss.item())
            '''status = '[{0}] loss = {1:0.5f} avg = {2:0.5f}, LR = {5:0.7f}'.format(
                epoch + 1, loss, np.mean(epoch_losses), scheduler.get_lr()[0])
            train_iterator.set_description(status)'''
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x), len(train_iterator.dataset),
                    100. * batch_idx / len(train_iterator), loss.item()))
            loss.backward()
            optimizer.step()
        scheduler.step()
        torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["PSPNet", str(epoch + 1)])))
        train_loss = np.mean(epoch_losses)
        writer.add_scalar('data/Train_loss', train_loss, epoch)
        print('Average loss: '+str(train_loss))
        
        # eval_data
        OA_all_1 = []
        OA_all_2 = []
        meanIU_all_1 = []
        meanIU_all_2 = []        
        real_ratios = []
        pred_ratios = []
        y_clss = []
        for batch_idx, (x, y, y_cls) in enumerate(val_iterator):
            x, y = utils.center_crop(x, y, crop)
            x, y, y_cls = Variable(x).cuda(), Variable(y).cuda(), Variable(y_cls).cuda()
            out, y_cls_pred = net(x)
            y_cls_pred = y_cls_pred.detach().cpu().numpy()
            y_clss.append(y_cls_pred)
            y = y.cpu().numpy()
            out_map = np.argmax(out.detach().cpu().numpy(), axis = 1)
            for gt, pred in zip(y, out_map):
                OA = utils.comp_OA(gt, pred)
                real_ratio, pred_ratio = utils.process_im(gt, pred)
                real_ratios.append(real_ratio)
                pred_ratios.append(pred_ratio)
                meanIU = utils.comp_meanIU(gt, pred)
                OA_all_1.append(OA[0])
                meanIU_all_1.append(meanIU[0])
                try:
                    OA_all_2.append(OA[1])
                    meanIU_all_2.append(meanIU[1])
                except:
                    OA_all_2.append(None)
                    meanIU_all_2.append(None)
        print('Val Eyelid Overall Accuracy: '+str(np.mean(OA_all_1)))
        print('Val Atrophy Overall Accuracy: '+str(utils.nanmean(OA_all_2)))
        print('Val Eyelid MeanIU: '+str(np.mean(meanIU_all_1)))
        print('Val Atrophy MeanIU: '+str(utils.nanmean(meanIU_all_2)))
        
        st()
        OA1, OA2, meanIU1, meanIU2, rmsd, msOA, msOA_avg = utils.process_add_metric(real_ratios, pred_ratios, OA_all_1, OA_all_2, meanIU_all_1, meanIU_all_2, names, np.concatenate(y_clss), threshold)
        
        writer.add_scalar('data/avg_meibo', msOA_avg, epoch)
        writer.add_scalar('data/meibo0', msOA[0], epoch)
        writer.add_scalar('data/meibo1', msOA[1], epoch)
        writer.add_scalar('data/meibo2', msOA[2], epoch)
        writer.add_scalar('data/meibo3', msOA[3], epoch)
        writer.add_scalar('data/OA_all_1', np.mean(OA_all_1), epoch)
        writer.add_scalar('data/OA_all_2', utils.nanmean(OA_all_2), epoch)
        writer.add_scalar('data/meanIU_all_1', np.mean(meanIU_all_1), epoch)
        writer.add_scalar('data/meanIU_all_2', utils.nanmean(meanIU_all_2), epoch)
        
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
        
if __name__ == '__main__':
    train()