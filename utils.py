import numpy as np
import scipy.misc
import torch
from pdb import set_trace as st
import h5py
import json
import ast
import imp
from sklearn.metrics import confusion_matrix

#threshold = 0 # 0.14
#threshold = 0.1

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 

def save_json(path, data):
    with open(path, 'w') as fp:
        json.dump(data, fp)
        
def load_json(path):
    with open(path, 'r') as fp:
        data = json.load(fp)
    return data

def process_add_metric(real_ratios, pred_ratios, OA_all_1, OA_all_2, meanIU_all_1, meanIU_all_2, names, y_clss, threshold):
    real_ratios = np.array(real_ratios)
    pred_ratios = np.array(pred_ratios)
    real_ms = ratio2mg(real_ratios)
    pred_ms = ratio2mg(pred_ratios, threshold)
    #ind  = pred_ratios<= threshold
    if threshold:
        ind = y_clss[:,1] < threshold
        pred_ratios[ind] = 0
    
    OA_all_1 = np.array(OA_all_1)
    OA_all_2 = np.array(OA_all_2)
    meanIU_all_1 = np.array(meanIU_all_1)
    meanIU_all_2 = np.array(meanIU_all_2)
    
    num0 = np.where(real_ms == 0)
    num1 = np.where(real_ms == 1)
    num2 = np.where(real_ms == 2)
    num3 = np.where(real_ms == 3)
    
    OA1 = [np.mean(OA_all_1[num0]), np.mean(OA_all_1[num1]),np.mean(OA_all_1[num2]),np.mean(OA_all_1[num3]) ]
    OA2 = [ nanmean(OA_all_2[num1]),nanmean(OA_all_2[num2]),nanmean(OA_all_2[num3]) ]

    meanIU1 = [np.mean(meanIU_all_1[num0]), np.mean(meanIU_all_1[num1]),np.mean(meanIU_all_1[num2]),np.mean(meanIU_all_1[num3]) ]
    meanIU2 = [nanmean(meanIU_all_2[num1]),nanmean(meanIU_all_2[num2]),nanmean(meanIU_all_2[num3]) ]
    rmsd_val = [rmsd( real_ratios[num0] , pred_ratios[num0] ), rmsd( real_ratios[num1] , pred_ratios[num1] ), rmsd( real_ratios[num2] , pred_ratios[num2] ), rmsd( real_ratios[num3] , pred_ratios[num3] )]
    
    ratings = load_json("/home/peterwg/dataset/meibo2018/rating_dic.json")
    
    # computer vs human
    tyms = []
    studyms = []
    real_ms_ = []
    pred_ms_ = []
    for i, name in enumerate (names):
        try:
            tyms.append(ratings[name]['tyms'])
            studyms.append(ratings[name]['studyms'])
            real_ms_.append(real_ms[i])
            pred_ms_.append(pred_ms[i])
        except:
            continue
    tyms = np.array(tyms)
    studyms = np.array(studyms)
    real_ms_ = np.array(real_ms_)
    pred_ms_ = np.array(pred_ms_)
    #print(np.mean(studyms == real_ms_), np.mean(tyms == real_ms_), np.mean(pred_ms_ == real_ms_) )
    
    num0 = np.where(real_ms_ == 0)
    num1 = np.where(real_ms_ == 1)
    num2 = np.where(real_ms_ == 2)
    num3 = np.where(real_ms_ == 3)    
    
    #msOA_study = [ np.mean(studyms[num0] == real_ms_[num0]), np.mean(studyms[num1] == real_ms_[num1]), np.mean(studyms[num2] == real_ms_[num2]), np.mean(studyms[num3] == real_ms_[num3]) ]
    #msOA_ty = [ np.mean(tyms[num0] == real_ms_[num0]), np.mean(tyms[num1] == real_ms_[num1]), np.mean(tyms[num2] == real_ms_[num2]), np.mean(tyms[num3] == real_ms_[num3]) ]
    msOA_comp = [np.mean(pred_ms_[num0] == real_ms_[num0]), np.mean(pred_ms_[num1] == real_ms_[num1]), np.mean(pred_ms_[num2] == real_ms_[num2]), np.mean(pred_ms_[num3] == real_ms_[num3]) ]
    print(msOA_comp, np.mean(pred_ms_ == real_ms_) )
    #print("Virtual vs human")
    #print(msOA_study, msOA_ty, msOA_comp)
    #print('Human confusion matrix')
    #print(confusion_matrix(studyms, tyms) )
    
    return (OA1, OA2, meanIU1, meanIU2, rmsd_val, msOA_comp, np.mean(pred_ms_ == real_ms_) )
    
def nanmean(lis):
    lis_ = [i for i in list(lis) if i is not None]
    return np.mean( np.array(lis_) )

def rmsd(real_ratios , pred_ratios):
    return np.sqrt(np.mean(( real_ratios - pred_ratios )**2))

def ratio2mg(ratio, TH = 0.0001):
    out = []
    for i in ratio:
        if i > 0.66:
            out.append(3)
        elif i >= 0.33:
            out.append(2)
        elif i > TH:
            out.append(1)
        else:
            out.append(0)
    return (np.array(out) )

def threshold_ratio(ratio, TH):
    for i, val in enumerate(ratio):
        if val < TH:
            ratio[i] = 0
    return ratio

def process_im(y_true, y_pred):
    _, count_t = np.unique(y_true, return_counts = True)
    _, count_p = np.unique(y_pred, return_counts = True)
    try:
        real_ratio = count_t[2]*1.0/ (count_t[2] + count_t[1]) 
    except IndexError:
        real_ratio = 0
    try:
        pred_ratio = count_p[2]*1.0/ (count_p[2] + count_p[1]) 
    except IndexError:
        pred_ratio = 0        
    
    return(real_ratio, pred_ratio) 
   
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
        IOU = overlap.sum()/(float(union.sum()) + 0.0000001)
        meanIU_all.append(IOU)
    return meanIU_all

def load_data(data_dir, batch_size, resize, sampler_defs=None, shuffle=True):
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
    if sampler_defs:
        sampler_dic = {'sampler': imp.load_source('', sampler_defs['def_file']).get_sampler(), 
                       'num_samples_cls': sampler_defs['num_samples_cls']}
    else:
        sampler_dic = None
    if sampler_dic is not None:
        train_loader_dataset = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle = False, sampler=sampler_dic['sampler'](sampler_dic['num_samples_cls']) )
    else:
        train_loader_dataset = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle = shuffle)        
    return train_loader_dataset

def load_data_nnames(data_dir, batch_size, resize, shuffle=True):
    train_data = h5py.File(data_dir , 'r')
    xs = np.array(train_data['xs'])
    ys = np.array(train_data['ys'])
    y_clss = np.array(train_data['y_clss'])
    names = np.array(train_data['names'])
    
    # decode metas
    names_ = []
    for i in names:
        names_.append( i.decode('UTF-8') )
        #metas_.append(ast.literal_eval(i.decode('UTF-8')) )
    # data_process
    xs = resize_im(xs, resize, resize)
    xs = norm_im(xs)
    ys = resize_mask(ys, resize, resize)
    y_clss = np.concatenate((y_clss, np.ones((len(y_clss),1))), axis=-1 )
            
    train_loader = torch.utils.data.TensorDataset(torch.from_numpy(xs).unsqueeze(1).type(torch.FloatTensor), torch.from_numpy(ys).type(torch.LongTensor), torch.from_numpy(y_clss).type(torch.FloatTensor))
    train_loader_dataset = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle = shuffle)
    return train_loader_dataset, names_ 
