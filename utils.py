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

def process_add_metric(names, real_ratios, pred_ratios, OA_all_1, OA_all_2, meanIU_all_1, meanIU_all_2, y_clss, mg_clss_gt, threshold= 0.05, confidence_th = 0.9):

    num0 = np.where(mg_clss_gt == 0)
    num1 = np.where(mg_clss_gt == 1)
    num2 = np.where(mg_clss_gt == 2)
    num3 = np.where(mg_clss_gt == 3)
    count_num = range(len(mg_clss_gt))
    
    real_ratios = np.array(real_ratios)
    pred_ratios = np.array(pred_ratios)
    
    pred_ms = ratio2mg(pred_ratios)
      
    #y_clss = np.exp(y_clss)
    #change_nd = np.where(ind)[0][np.max(y_clss, axis=1)[ind]> confidence_th]
    #f_clss_eval = np.argmax(f_clss, axis = 1)
    #mg_clss = np.argmax(mg_clss, axis = 1)
    '''if threshold:
        pred_ratios[y_clss == 0] = 0'''
    
    
    '''ori_acc = [np.mean(pred_ms[num0] == mg_clss_gt[num0]), np.mean(pred_ms[num1] == mg_clss_gt[num1]), np.mean(pred_ms[num2] == mg_clss_gt[num2]), np.mean(pred_ms[num3] == mg_clss_gt[num3]), np.mean(pred_ms == mg_clss_gt)]
    
    for i in change_nd:
        pred_ms[i] = y_clss_eval[i]'''
    #new_acc = [np.mean(pred_ms[num0] == mg_clss_gt[num0]), np.mean(pred_ms[num1] == mg_clss_gt[num1]), np.mean(pred_ms[num2] == mg_clss_gt[num2]), np.mean(pred_ms[num3] == mg_clss_gt[num3]), np.mean(pred_ms == mg_clss_gt)]
    #new_acc = [np.mean(f_clss_eval[num0] == mg_clss_gt[num0]), np.mean(f_clss_eval[num1] == mg_clss_gt[num1]), np.mean(f_clss_eval[num2] == mg_clss_gt[num2]), np.mean(f_clss_eval[num3] == mg_clss_gt[num3]), np.mean(f_clss_eval == mg_clss_gt)]
    
    #print(new_acc)
    
    OA_all_1 = np.array(OA_all_1)
    OA_all_2 = np.array(OA_all_2)
    meanIU_all_1 = np.array(meanIU_all_1)
    meanIU_all_2 = np.array(meanIU_all_2)
    
    #classification_acc = [np.mean(mg_clss[num0] == mg_clss_gt[num0]), np.mean(mg_clss[num1] == mg_clss_gt[num1]), np.mean(mg_clss[num2] == mg_clss_gt[num2]), np.mean(mg_clss[num3] == mg_clss_gt[num3]), np.mean(mg_clss == mg_clss_gt)]
    
    OA1 = [np.mean(OA_all_1[num0]), np.mean(OA_all_1[num1]),np.mean(OA_all_1[num2]),np.mean(OA_all_1[num3]) ]
    OA2 = [ nanmean(OA_all_2[num1]),nanmean(OA_all_2[num2]),nanmean(OA_all_2[num3]) ]

    meanIU1 = [np.mean(meanIU_all_1[num0]), np.mean(meanIU_all_1[num1]),np.mean(meanIU_all_1[num2]),np.mean(meanIU_all_1[num3]) ]
    meanIU2 = [nanmean(meanIU_all_2[num1]),nanmean(meanIU_all_2[num2]),nanmean(meanIU_all_2[num3]) ]
    rmsd_val = [rmsd( real_ratios[num0] , pred_ratios[num0] ), rmsd( real_ratios[num1] , pred_ratios[num1] ), rmsd( real_ratios[num2] , pred_ratios[num2] ), rmsd( real_ratios[num3] , pred_ratios[num3] )]
    
    ratings = load_json("/home/peterwg/dataset/meibo2018/rating_dic.json")
    
    # computer vs human
    ty_acc_ = []
    study_acc_ = []
    alg_acc_ = []
    
    for threshold in [0, 0.0025]:#np.arange(0.00025, 0.05, 0.00025):
        ind = ( (real_ratios> 0) & (real_ratios< threshold) ) | ((real_ratios> 0.33-threshold) & (real_ratios< 0.33+threshold)) |  ((real_ratios> 0.66-threshold) & (real_ratios< 0.66+threshold))
        ind0 = (real_ratios> 0-threshold) & (real_ratios< threshold)
        ind1 = (real_ratios> 0.33-threshold) & (real_ratios< 0.33+threshold)
        ind2 = (real_ratios> 0.66-threshold) & (real_ratios< 0.66+threshold)
        
        tyms = []
        studyms = []
        real_ms_ = []
        pred_ms_ = []
        ind0_ = []
        ind1_ = []
        ind2_ = []
        count_num_ = []
        real_ratios_ = []
        pred_ratios_ = []
        for i, name in enumerate (names):
            try:
                tyms.append(ratings[name]['tyms'])
                studyms.append(ratings[name]['studyms'])
                real_ms_.append(mg_clss_gt[i])
                pred_ms_.append(pred_ms[i])
                ind0_.append(ind0[i])
                ind1_.append(ind1[i])
                ind2_.append(ind2[i])
                count_num_.append(count_num[i])
                real_ratios_.append(real_ratios[i])
                pred_ratios_.append(pred_ratios[i])
            except:
                continue
        tyms = np.array(tyms)
        studyms = np.array(studyms)
        real_ms_ = np.array(real_ms_)
        pred_ms_ = np.array(pred_ms_)
        real_ratios_ = np.array(real_ratios_)
        pred_ratios_ = np.array(pred_ratios_)
        #print(np.mean(studyms == real_ms_), np.mean(tyms == real_ms_), np.mean(pred_ms_ == real_ms_) )
        
        
        #msOA_study = [ np.mean(studyms[num0] == real_ms_[num0]), np.mean(studyms[num1] == real_ms_[num1]), np.mean(studyms[num2] == real_ms_[num2]), np.mean(studyms[num3] == real_ms_[num3]) ]
        #msOA_ty = [ np.mean(tyms[num0] == real_ms_[num0]), np.mean(tyms[num1] == real_ms_[num1]), np.mean(tyms[num2] == real_ms_[num2]), np.mean(tyms[num3] == real_ms_[num3]) ]
        #msOA_comp = [np.mean(pred_ms_[num0] == real_ms_[num0]), np.mean(pred_ms_[num1] == real_ms_[num1]), np.mean(pred_ms_[num2] == real_ms_[num2]), np.mean(pred_ms_[num3] == real_ms_[num3]) ]
        #print(msOA_comp, np.mean(pred_ms_ == real_ms_) )
        #print("Virtual vs human")
        #print(msOA_study, msOA_ty, msOA_comp)
        #print('Human confusion matrix')
        #print(confusion_matrix(studyms, tyms) )
        ty_acc = compute_fuzz_acc(tyms, real_ms_, ind0_, ind1_, ind2_)
        study_acc = compute_fuzz_acc(studyms, real_ms_, ind0_, ind1_, ind2_)
        alg_acc = compute_fuzz_acc(pred_ms_ ,real_ms_, ind0_, ind1_, ind2_)
        
        ty_acc_.append(ty_acc)
        study_acc_.append(study_acc)
        alg_acc_.append(alg_acc)
    
    ty_acc_ = np.array(ty_acc_)
    study_acc_ = np.array(study_acc_)
    alg_acc_ = np.array(alg_acc_)

    count_num_ = np.array(count_num_)
    algW = [count_num_[pred_ms_ != real_ms_], pred_ratios_[pred_ms_ != real_ms_], real_ms_[pred_ms_ != real_ms_], pred_ms_[pred_ms_ != real_ms_], tyms[pred_ms_ != real_ms_], studyms[pred_ms_ != real_ms_] ]
    tyW = [count_num_[tyms != real_ms_], pred_ratios_[tyms != real_ms_], real_ms_[tyms != real_ms_], pred_ms_[tyms != real_ms_], tyms[tyms != real_ms_], studyms[tyms != real_ms_] ]
    print("Alg wrong****")
    print(algW[0])
    print(algW[1])
    print(algW[2])
    print(algW[3])
    print(algW[4])
    print(algW[5])
    print("TY wrong****")
    print(tyW[0])
    print(tyW[1])
    print(tyW[2])
    print(tyW[3]) 
    print(tyW[4]) 
    print(tyW[5]) 
    
    return (OA1, OA2, meanIU1, meanIU2, rmsd_val) #msOA_comp, np.mean(pred_ms_ == real_ms_) )
    
def compute_fuzz_acc(pred_ms_, real_ms_, ind0_, ind1_, ind2_):
    num_test = len(real_ms_)
    rest = (1- (np.array(ind0_) | np.array(ind1_) |np.array(ind2_)) ).astype(bool)
    interval = np.sum((pred_ms_[rest] == real_ms_[rest]))
    
    interval0 = np.sum((pred_ms_[rest] == real_ms_[rest])[real_ms_[rest] == 0])
    interval1 = np.sum((pred_ms_[rest] == real_ms_[rest])[real_ms_[rest] == 1])
    interval2 = np.sum((pred_ms_[rest] == real_ms_[rest])[real_ms_[rest] == 2])
    interval3 = np.sum((pred_ms_[rest] == real_ms_[rest])[real_ms_[rest] == 3])
    
    th0_0 = np.sum((np.abs(real_ms_[ind0_] - pred_ms_[ind0_]) <=1)[real_ms_[ind0_] == 0])
    th0_1 = np.sum((np.abs(real_ms_[ind0_] - pred_ms_[ind0_]) <=1)[real_ms_[ind0_] == 1])
    th1_1 = np.sum((np.abs(real_ms_[ind1_] - pred_ms_[ind1_]) <=1)[real_ms_[ind1_] == 1])
    th1_2 = np.sum((np.abs(real_ms_[ind1_] - pred_ms_[ind1_]) <=1)[real_ms_[ind1_] == 2])
    th2_2 = np.sum((np.abs(real_ms_[ind2_] - pred_ms_[ind2_]) <=1)[real_ms_[ind2_] == 2])
    th2_3 = np.sum((np.abs(real_ms_[ind2_] - pred_ms_[ind2_]) <=1)[real_ms_[ind2_] == 3])    

    acc0 = (interval0+th0_0)*1.0 /(np.sum(real_ms_==0))
    acc1 = (interval1+th0_1+th1_1)*1.0 /(np.sum(real_ms_==1))
    acc2 = (interval2+th1_2+th2_2)*1.0 /(np.sum(real_ms_==2))
    acc3 = (interval3+th2_3)*1.0 /(np.sum(real_ms_==3))
    
    return np.array([acc0, acc1, acc2, acc3, (interval+th0_0+th0_1+th1_1+th1_2+th2_2+th2_3)*1.0/num_test] )
    
    
def nanmean(lis):
    lis_ = [i for i in list(lis) if i is not None]
    return np.mean( np.array(lis_) )

def rmsd(real_ratios , pred_ratios):
    return np.sqrt(np.mean(( real_ratios - pred_ratios )**2))

def ratio2mg(ratio, TH = 0.001):
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
    ms = np.array(train_data['ms'])
    # data_process
    xs = resize_im(xs, resize, resize)
    xs = norm_im(xs)
    ys = resize_mask(ys, resize, resize)
    y_clss = np.concatenate((y_clss, np.ones((len(y_clss),1))), axis=-1 )
            
    train_loader = torch.utils.data.TensorDataset(torch.from_numpy(xs).unsqueeze(1).type(torch.FloatTensor), torch.from_numpy(ys).type(torch.LongTensor), torch.from_numpy(y_clss[:,1]).type(torch.LongTensor), torch.from_numpy(ms).type(torch.LongTensor))
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
    ms = np.array(train_data['ms'])
    
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
            
    train_loader = torch.utils.data.TensorDataset(torch.from_numpy(xs).unsqueeze(1).type(torch.FloatTensor), torch.from_numpy(ys).type(torch.LongTensor), torch.from_numpy(y_clss[:,1]).type(torch.LongTensor), torch.from_numpy(ms).type(torch.LongTensor))
    train_loader_dataset = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle = shuffle)
    return train_loader_dataset, names_ 
