import numpy as np
import scipy.misc
import torch
from pdb import set_trace as st
import h5py

def resize_im(ims, resizeH, resizeW):
    temp = []
    for im in ims:
        temp.append(scipy.misc.imresize(im, (resizeH, resizeW)) )
    return np.array(temp)

def resize_singleim(im, resizeH, resizeW):
    return scipy.misc.imresize(im, (resizeH, resizeW))
    
def random_crop(xs, ys, crop):
    sizea = xs.shape[-1]
    start = np.random.randint(sizea-crop)
    return xs[..., start:start+crop, start:start+crop], ys[..., start:start+crop, start:start+crop]

def center_crop(xs, ys, crop):
    sizea = xs.shape[-1]
    start = int(0.5*(sizea-crop))
    return xs[..., start:start+crop, start:start+crop], ys[..., start:start+crop, start:start+crop]

def resize_mask(ims, resizeH, resizeW):
    temp = []
    for im in ims:
        temp.append(scipy.misc.imresize(im, (resizeH, resizeW) , interp = 'nearest', mode = 'F') )
    return np.array(temp) #np.rint(np.array(temp)/128.0)

def norm_im(im):
    im = im*1.0/255
    im = 2*im - 1
    return im

def unnorm_im(im):
    im = np.squeeze(im)
    im = (im + 1.0)/2
    im = im
    return im.astype(float)

def comp_OA(y_true, y_pred):
    class_num = np.max( np.unique(y_true) ) + 1
    OA_all = []
    for i in range(1, class_num):
        if i == 1:
            y_true_temp = (y_true != 0 ).astype(int)
            y_pred_temp = (y_pred != 0 ).astype(int)
        else:
            y_true_temp = (y_true==i).astype(int)
            y_pred_temp = (y_pred==i).astype(int)
        OA_all.append(np.mean(y_true_temp == y_pred_temp))
    return OA_all

def comp_meanIU(y_true, y_pred):
    class_num = np.max( np.unique(y_true) ) + 1
    meanIU_all = []
    for i in range(1, class_num):
        if i == 1:
            y_true_temp = (y_true != 0 )
            y_pred_temp = (y_pred != 0 )
        else:
            y_true_temp = (y_true==i)
            y_pred_temp = (y_pred==i)
        overlap = y_true_temp*y_pred_temp # Logical AND
        union = y_true_temp + y_pred_temp # Logical OR
        IOU = overlap.sum()/float(union.sum())
        meanIU_all.append(IOU)
    return meanIU_all

def load_data(data_dir, batch_size, resize, shuffle=True):
    train_data = h5py.File(data_dir , 'r')
    xs = np.array(train_data['xs'])
    ys = np.array(train_data['ys'])
    y_clss = np.array(train_data['y_clss'])
    # data_process
    xs = resize_im(xs, resize, resize)
    xs = norm_im(xs)
    ys = resize_mask(ys, resize, resize)
    y_clss = np.concatenate((y_clss, np.ones((len(y_clss),1))), axis=-1 )
            
    train_loader = torch.utils.data.TensorDataset(torch.from_numpy(xs).unsqueeze(1).type(torch.FloatTensor), torch.from_numpy(ys).type(torch.LongTensor), torch.from_numpy(y_clss).type(torch.FloatTensor))
    train_loader_dataset = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle = shuffle)
    return train_loader_dataset