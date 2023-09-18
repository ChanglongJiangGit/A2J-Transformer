import cv2
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import scipy.io as scio
import os
from PIL import Image
import time
import datetime
import logging
from torch.autograd import Variable
import sys
# import model_depthreg_noncomp as model
# import anchor_depthreg_noncomp  as anchor
from tqdm import tqdm
import random_erasing


from model import get_model
from torch.nn.parallel.data_parallel import DataParallel

from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


save_dir = '/data/data2/jiangchanglong/a2jformer/code3xyz_hands17/output_hands2017/model_dump_new_depth'
model_dir = '/data/data2/jiangchanglong/a2jformer/code3xyz_hands17/output_hands2017/model_dump_new_depth/net_0_wetD_0.0001_depFact_0.5_RegFact_3_rndShft_5.pth'
tensorboard_dir = '/data/data2/jiangchanglong/a2jformer/code3xyz_hands17/output_hands2017/tensorboard_log'
'''
commend for tensorboard is : 
tensorboard --logdir=/data/data2/jiangchanglong/a2jformer/code3xyz_hands17/output_hands2017/tensorboard_log
'''
# DataHyperParms 
#TrainImgFrames = 22067
TestImgFrames = 295510

validIndex_train = np.load('/data/data1/zhangboshen/CODE/219_A2J_original/Anchor_Pose_fpn/data/Hands2017/validIndex_955101.npy')
validIndex_test = np.arange(TestImgFrames)
TrainImgFrames = len(validIndex_train)
keypointsNumber = 21
cropWidth = 176
cropHeight = 176

batch_size = 64
# learning_rate = 0.00035
lr = 1e-4
# lr_dec_epoch = [8,11,14]
# lr_dec_factor = 5

Weight_Decay = 1e-4
nepoch = 17

RegLossFactor = 3
spatialFactor = 0.5
RandCropShift = 5
RandshiftDepth = 1
RandRotate = 180 ## +- 30
RandScale = (1.0, 0.5)
xy_thres = 100
depth_thres = 150
from shutil import copyfile




try:
    os.makedirs(save_dir)
except OSError:
    pass

trainingImageDir = '/data/data1/zhangboshen/CODE/219_A2J_original/Anchor_Pose_fpn/data/Hands2017/images/'
testingImageDir = '/data/data1/zhangboshen/CODE/219_A2J_original/Anchor_Pose_fpn/data/Hands2017/frame/images/'

test_annotation_dir = '/data/data2/jiangchanglong/test_annotation_frame.txt'

def pixel2world(x, fx, fy, ux, uy):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x


def world2pixel(x, fx, fy, ux, uy):
    x[:, :, 0] = x[:, :, 0] * fx / x[:, :, 2] + ux
    x[:, :, 1] = x[:, :, 1] * fy / x[:, :, 2] + uy
    return x
    

# loading labels
print('loading keypoints labels')
keypointsfile = '/data/data1/zhangboshen/CODE/219_A2J_original/Anchor_Pose_fpn/data/Hands2017/train_keypointsUVD.mat'
keypoints_mat = scio.loadmat(keypointsfile)
keypointsUVD_train = keypoints_mat['keypoints3D']   
keypointsUVD_train = keypointsUVD_train.astype(np.float32)
print('training keypointsWorldtest_shape',np.shape(keypointsUVD_train))

# loading labels
# print('loading keypoints labels (testingImages)')
# keypointsfileTest = '/data/zhangboshen/CODE/Anchor_Pose_fpn/data/ICVL/test_keypointsUVD.mat'
# keypoints_matTest = scio.loadmat(keypointsfileTest)
# keypointsUVD_test = keypoints_matTest['keypoints3D']   
# keypointsUVD_test = keypointsUVD_test.astype(np.float32)
# print('testing keypointsWorldtest_shape',np.shape(keypointsUVD_test))

########## This label will not be used during evaluation
keypointsUVD_test = keypointsUVD_train[:TestImgFrames]


print('loading center labels')
center_train = scio.loadmat('/data/data1/zhangboshen/CODE/219_A2J_original/Anchor_Pose_fpn/data/Hands2017/train_centre_pixel.mat')['centre_pixel']
center_train = center_train.astype(np.float32)
print('center_train_shape',np.shape(center_train))

centre_train_world = pixel2world(center_train.copy(), 475.065948, 475.065857, 315.944855, 245.287079)

centerlefttop_train = centre_train_world.copy()
centerlefttop_train[:,0,0] = centerlefttop_train[:,0,0]-xy_thres
centerlefttop_train[:,0,1] = centerlefttop_train[:,0,1]+xy_thres

centerrightbottom_train = centre_train_world.copy()
centerrightbottom_train[:,0,0] = centerrightbottom_train[:,0,0]+xy_thres
centerrightbottom_train[:,0,1] = centerrightbottom_train[:,0,1]-xy_thres

train_lefttop_pixel = world2pixel(centerlefttop_train, 475.065948, 475.065857, 315.944855, 245.287079)
train_rightbottom_pixel = world2pixel(centerrightbottom_train, 475.065948, 475.065857, 315.944855, 245.287079)


center_test = scio.loadmat('/data/data1/zhangboshen/CODE/219_A2J_original/Anchor_Pose_fpn/data/Hands2017/test_centre_pixel.mat')['centre_pixel']
center_test = center_test.astype(np.float32)
print('center_test_shape',np.shape(center_test))

centre_test_world = pixel2world(center_test.copy(), 475.065948, 475.065857, 315.944855, 245.287079)

centerlefttop_test = centre_test_world.copy()
centerlefttop_test[:,0,0] = centerlefttop_test[:,0,0]-xy_thres
centerlefttop_test[:,0,1] = centerlefttop_test[:,0,1]+xy_thres

centerrightbottom_test = centre_test_world.copy()
centerrightbottom_test[:,0,0] = centerrightbottom_test[:,0,0]+xy_thres
centerrightbottom_test[:,0,1] = centerrightbottom_test[:,0,1]-xy_thres

test_lefttop_pixel = world2pixel(centerlefttop_test, 475.065948, 475.065857, 315.944855, 245.287079)
test_rightbottom_pixel = world2pixel(centerrightbottom_test, 475.065948, 475.065857, 315.944855, 245.287079)


    
def transform(img, label, matrix):
    '''
    img: H, W,  label, 14,2,   
    '''
    img_out = cv2.warpAffine(img,matrix,(cropWidth,cropHeight))
    label_out = np.ones((keypointsNumber, 3))
    label_out[:,:2] = label[:,:2].copy()
    label_out = np.matmul(matrix, label_out.transpose())
    label_out = label_out.transpose()

    return img_out, label_out


def dataPreprocess(index, img, keypointsUVD, center, mean, std, lefttop_pixel, rightbottom_pixel, validIndex, xy_thres=100, depth_thres=150, augment=True):
    '''
    img [H, W], 
    '''
 
    imageOutputs = np.ones((cropHeight, cropWidth, 1), dtype='float32') 
    labelOutputs = np.ones((keypointsNumber, 3), dtype = 'float32') 
    

    if augment:
        RandomOffset_1 = np.random.randint(-1*RandCropShift,RandCropShift)
        RandomOffset_2 = np.random.randint(-1*RandCropShift,RandCropShift)
        RandomOffset_3 = np.random.randint(-1*RandCropShift,RandCropShift)
        RandomOffset_4 = np.random.randint(-1*RandCropShift,RandCropShift)
        #RandomOffsetDepth = np.random.randint(-1*RandshiftDepth,RandshiftDepth)
        #RandomOffsetDepth = (np.random.rand(cropHeight,cropWidth)-0.5)*RandshiftDepth
        RandomOffsetDepth = np.random.normal(0, RandshiftDepth, cropHeight*cropWidth).reshape(cropHeight,cropWidth) 
        RandomOffsetDepth[np.where(RandomOffsetDepth < RandshiftDepth)] = 0
        RandomRotate = np.random.randint(-1*RandRotate,RandRotate)
        RandomScale = np.random.rand()*RandScale[0]+RandScale[1]
        #RandomScaleDepth = np.random.rand()*RandScale[0]+RandScale[1]
        matrix = cv2.getRotationMatrix2D((cropWidth/2,cropHeight/2),RandomRotate,RandomScale)

    else:
        RandomOffset_1, RandomOffset_2, RandomOffset_3, RandomOffset_4 = 0, 0, 0, 0
        RandomRotate = 0
        RandomScale = 1
        RandomOffsetDepth = 0
        #RandomScaleDepth = 1
        matrix = cv2.getRotationMatrix2D((cropWidth/2,cropHeight/2),RandomRotate,RandomScale)

    # new_Xmin = center_pixel[index][0][0] - 50//2 
    # new_Ymin = center_pixel[index][0][1] - 50//2 
    # new_Xmax = center_pixel[index][0][0] + 50//2 
    # new_Ymax = center_pixel[index][0][1] + 50//2 
 
    new_Xmin = max(lefttop_pixel[index,0,0] + RandomOffset_1, 0)
    new_Ymin = max(rightbottom_pixel[index,0,1] + RandomOffset_2, 0)  
    new_Xmax = min(rightbottom_pixel[index,0,0] + RandomOffset_3, img.shape[1] - 1)
    new_Ymax = min(lefttop_pixel[index,0,1] + RandomOffset_4, img.shape[0] - 1)

    #imCrop = img.copy()[int(lefttop_pixel[index,0,1]) : int(rightbottom_pixel[index,0,1]), \
                #int(lefttop_pixel[index,0,0]) : int(rightbottom_pixel[index,0,0])]
    imCrop = img.copy()[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)]

    imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)

    imgResize = np.asarray(imgResize,dtype = 'float32')  # H*W*C

    imgResize[np.where(imgResize >= center[index][0][2] + depth_thres)] = center[index][0][2] #+ depth_thres#885.4240396601466
    imgResize[np.where(imgResize <= center[index][0][2] - depth_thres)] = center[index][0][2] #- depth_thres#885.4240396601466
    imgResize = (imgResize - center[index][0][2])*RandomScale

    #imgResize += RandomOffsetDepth

    imgResize = (imgResize - mean) / std


    #imgResize *= RandomScaleDepth
#######################
    #imgResize[np.where(imgResize >= center_pixel[index][0][2] + depth_thres)] = center_pixel[index][0][2] + depth_thres#885.4240396601466
    #imgResize[np.where(imgResize <= center_pixel[index][0][2] - depth_thres)] = center_pixel[index][0][2] - depth_thres#885.4240396601466
#######################
    #depthMin = np.min(imgResize)
    #depthMax = np.max(imgResize)
    #imgResize = 2*((imgResize-depthMin)/(depthMax-depthMin) - 0.5)  # normalize
    
    ## label
    label_xy = np.ones((keypointsNumber, 2), dtype = 'float32') 
    
    label_xy[:,0] = (keypointsUVD[validIndex[index],:,0].copy() - new_Xmin)*cropWidth/(new_Xmax - new_Xmin) # x
    label_xy[:,1] = (keypointsUVD[validIndex[index],:,1].copy() - new_Ymin)*cropHeight/(new_Ymax - new_Ymin) # y
    
    if augment:
        #imgResize += RandomOffsetDepth/depthFactor    
        imgResize, label_xy = transform(imgResize, label_xy, matrix)  ## rotation, scaling
    
    imageOutputs[:,:,0] = imgResize

    # labelOutputs[:,1] = label_xy[:,0]#(self.keypointsUVD[index,:,0] - new_Xmin)*cropWidth/(new_Xmax - new_Xmin) # x
    # labelOutputs[:,0] = label_xy[:,1] #(self.keypointsUVD[index,:,1] - new_Ymin)*cropHeight/(new_Ymax - new_Ymin) # y
    labelOutputs[:,1] = label_xy[:,1]#(self.keypointsUVD[index,:,0] - new_Xmin)*cropWidth/(new_Xmax - new_Xmin) # x
    labelOutputs[:,0] = label_xy[:,0] #(self.keypointsUVD[index,:,1] - new_Ymin)*cropHeight/(new_Ymax - new_Ymin) # y
    #labelOutputs[:,2] = (keypointsUVD.copy()[index,:,2])   # Z  
    labelOutputs[:,2] = (keypointsUVD[validIndex[index],:,2] - center[index][0][2])*RandomScale   # Z  
    #labelOutputs[:,2] *= RandomScaleDepth
      
    #labelOutputs[:,2] = 2*((labelOutputs[:,2]-depthMin)/(depthMax-depthMin) - 0.5)  # normalize
    
    imageOutputs = np.asarray(imageOutputs)
    imageNCHWOut = imageOutputs.transpose(2, 0, 1)  # [H, W, C] --->>>  [C, H, W]
    imageNCHWOut = np.asarray(imageNCHWOut)
    labelOutputs = np.asarray(labelOutputs)

    data, label = torch.from_numpy(imageNCHWOut), torch.from_numpy(labelOutputs)

    return data, label


######################   Pytorch dataloader   #################
class my_dataloader(torch.utils.data.Dataset):

    def __init__(self, trainingImageDir, center, lefttop_pixel, rightbottom_pixel, keypointsUVD, validIndex, augment=True):

        self.trainingImageDir = trainingImageDir
        self.mean = np.load('/data/data1/zhangboshen/CODE/219_A2J_original/Anchor_Pose_fpn/data/Hands2017/xy100_depz150_mean.npy')
        self.std = np.load('/data/data1/zhangboshen/CODE/219_A2J_original/Anchor_Pose_fpn/data/Hands2017/xy100_depz150_std.npy')
        print(self.mean, self.std)
        self.center = center
        self.lefttop_pixel = lefttop_pixel
        self.rightbottom_pixel = rightbottom_pixel
        self.keypointsUVD = keypointsUVD
        self.validIndex = validIndex
        self.augment = augment
        self.xy_thres = xy_thres
        self.depth_thres = depth_thres
        self.randomErase = random_erasing.RandomErasing(probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0], scale=1)
        self.bbox_3d_size = 400
        self.output_hm_shape = 64

    def __getitem__(self, index):

        #depth = scio.loadmat(self.trainingImageDir + str(self.validIndex[index]+1) + '.mat')['img']      
        depth = Image.open(self.trainingImageDir + 'image_D%.8d'%(self.validIndex[index]+1) + '.png') 
        depth = np.array(depth)
        #torch.cuda.synchronize() 
         
        data, label = dataPreprocess(index, depth, self.keypointsUVD, self.center, self.mean, self.std, \
            self.lefttop_pixel, self.rightbottom_pixel, self.validIndex, self.xy_thres, self.depth_thres, self.augment)
        #label[:,2] = 2*((label[:,2]-depthMin)/(depthMax-depthMin) - 0.5)
        #torch.cuda.synchronize() 

        if self.augment:
            data = self.randomErase(data)
            #data += RandomOffsetDepth
            #labelOutputs[:,2] += RandomOffsetDepth 

        # use zero mask for now. Later if required put ones along padded pixels
        mask = np.zeros((data.shape[1], data.shape[2])).astype(bool)

        joint_valid = np.ones(keypointsNumber)
        # label[:,2] = (label[:,2] / (self.bbox_3d_size/2) + 1)/2. * self.output_hm_shape
        joint_valid = joint_valid * ((label[:,0] >= 0) * (label[:,0] < 176)).numpy()
        joint_valid = joint_valid * ((label[:,1] >= 0) * (label[:,1] < 176)).numpy()
        # joint_valid = joint_valid * ((label[:,2] >= 0) * (label[:,2] < 64 )).numpy()
    
        inputs = {'img': data, 'mask': mask} 
        targets = {'joint_coord': label}
        meta_info = {'joint_valid': joint_valid}

        # return data, label
        return inputs, targets, meta_info
    
    def __len__(self):
        return len(self.center)

      
train_image_datasets = my_dataloader(trainingImageDir, center_train, train_lefttop_pixel, train_rightbottom_pixel, keypointsUVD_train, validIndex_train, augment=True)
train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size = batch_size,
                                             shuffle = True, num_workers = 8, drop_last=True)

val_image_datasets = my_dataloader(trainingImageDir, center_train, train_lefttop_pixel, train_rightbottom_pixel, keypointsUVD_train, validIndex_train, augment=False)
val_dataloaders = torch.utils.data.DataLoader(val_image_datasets, batch_size = batch_size,
                                             shuffle = False, num_workers = 8)#, collate_fn = my_collate_fn) # 8 workers may work faster

test_image_datasets = my_dataloader(testingImageDir, center_test, test_lefttop_pixel, test_rightbottom_pixel, keypointsUVD_test, validIndex_test, augment=False)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size = batch_size,
                                             shuffle = False, num_workers = 32)#, collate_fn = my_collate_fn) # 8 workers may work faster

def train():
    # net = model.Cls_Reg_Dual_Path_Net(num_classes = keypointsNumber)
    # #net.load_state_dict(torch.load('/data/zhangboshen/CODE/Anchor_Pose_fpn/Results_nyu/scale30_160160_xy110_d150/net_5_wetD_0.0001_depFact_0.5_RegFact_3_rndShft_5.pth'))
    # net = net.cuda()
    model = get_model('train', keypointsNumber)
    model = DataParallel(model).cuda()
    tbwriter = SummaryWriter(tensorboard_dir)
    
    # post_precess = anchor.post_process(shape=[cropHeight//16,cropWidth//16],stride=16,P_h=None, P_w=None)
    # criterion = anchor.FocalLoss(shape=[cropHeight//16,cropWidth//16],thres = [16.0,32.0],stride=16,\
    #     spatialFactor=spatialFactor,img_shape=[cropHeight, cropWidth],P_h=None, P_w=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=Weight_Decay)
    #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=Weight_Decay,momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                    filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
    logging.info('======================================================')

    for epoch in range(nepoch):

        model = model.train()
        scheduler.step(epoch)
        train_loss_add = 0.0
        Cls_loss_add = 0.0
        Reg_loss_add = 0.0
        timer = time.time()

        # for e in lr_dec_epoch:
        #     if epoch < e:
        #         break
        # if epoch < lr_dec_epoch[-1]:
        #     idx = lr_dec_epoch.index(e)
        #     for g in optimizer.param_groups:
        #         g['lr'] = lr / (lr_dec_factor ** idx)
        # else:
        #     for g in optimizer.param_groups:
        #         g['lr'] = lr / (lr_dec_factor ** len(lr_dec_epoch))

        # Training loop
        for i, (inputs, targets, meta_info) in enumerate(train_dataloaders):

            #torch.cuda.synchronize() 

            # img, label = img.cuda(), label.cuda()     
            # heads  = net(img)  
            inputs = {k:inputs[k].cuda() for k in inputs} 
            targets = {k:targets[k].cuda() for k in targets} 
            meta_info = {k:meta_info[k].cuda() for k in meta_info} 

            loss = model(inputs, targets, meta_info, 'train')
            loss = {k:loss[k].mean() for k in loss}
            
            Cls_loss, Reg_loss, total_loss = loss['Cls_loss'], loss['Reg_loss'], loss['total_loss']

            optimizer.zero_grad()  
            
            # Cls_loss, Reg_loss = criterion(heads, label)
            loss['total_loss'].backward()

            # loss = 1*Cls_loss + Reg_loss*RegLossFactor
            # loss.backward()
            optimizer.step()

            
            train_loss_add = train_loss_add + (total_loss.item())*len(inputs)
            Cls_loss_add = Cls_loss_add + (Cls_loss.item())*len(inputs)
            Reg_loss_add = Reg_loss_add + (Reg_loss.item())*len(inputs)

            for g in optimizer.param_groups:
                cur_lr = g['lr']
            # printing loss info
            if i%10 == 0:
                print('epoch: ',epoch,'lr: ',cur_lr, ' step: ', i, 'Cls_loss ',Cls_loss.item(), 'Reg_loss ',Reg_loss.item(), ' total loss ',total_loss.item())
                tbwriter.add_scalar('loss/total_loss', loss['total_loss'], epoch*len(train_dataloaders)+i)
                tbwriter.add_scalar('loss/class_loss', loss['Cls_loss'], epoch*len(train_dataloaders)+i)
                tbwriter.add_scalar('loss/regre_loss', loss['Reg_loss'], epoch*len(train_dataloaders)+i)
            tbwriter.add_scalar('learning_rate', cur_lr, epoch)
        scheduler.step(epoch)
            
        # time taken
        torch.cuda.synchronize()
        timer = time.time() - timer
        timer = timer / TrainImgFrames
        print('==> time to learn 1 sample = %f (ms)' %(timer*1000))


        train_loss_add = train_loss_add / TrainImgFrames
        Cls_loss_add = Cls_loss_add / TrainImgFrames
        Reg_loss_add = Reg_loss_add / TrainImgFrames
        print('mean train_loss_add of 1 sample: %f, #train_indexes = %d' %(train_loss_add, TrainImgFrames))
        print('mean Cls_loss_add of 1 sample: %f, #train_indexes = %d' %(Cls_loss_add, TrainImgFrames))
        print('mean Reg_loss_add of 1 sample: %f, #train_indexes = %d' %(Reg_loss_add, TrainImgFrames))


        Error_test = 0
        Error_train = 0

        if (epoch % 1 == 0):  
            # net = net.eval()
            model = model.eval()
            output = torch.FloatTensor()
            outputTrain = torch.FloatTensor()

            with torch.no_grad():   
                for i, (inputs, targets, meta_info) in tqdm(enumerate(test_dataloaders)):
                    # img, label = img.cuda(), label.cuda()       
                    # heads = net(img)  
                    # pred_keypoints = post_precess(heads, voting=False)
                    inputs = {k:inputs[k].cuda() for k in inputs} 
                    targets = {k:targets[k].cuda() for k in targets} 
                    meta_info = {k:meta_info[k].cuda() for k in meta_info}
                    out = model(inputs, targets, meta_info, 'test')
                    pred_keypoints = out['joint_coord'] 
                    output = torch.cat([output,pred_keypoints.data.cpu()], 0)

            result = output.cpu().data.numpy()  # N, 21, 3
            Error_test = errorCompute(result,keypointsUVD_test, center_test)
            print('epoch: ', epoch, 'Test error:', Error_test)
            # scio.savemat(os.path.join(save_dir, 'epoch_' + str(epoch) + '.mat'),{'result':result})
            # writeTxt(result, center_test, epoch)
            saveNamePrefix = '%s/net_%d_wetD_' % (save_dir, epoch) + str(Weight_Decay) + '_depFact_' + str(spatialFactor) + '_RegFact_' + str(RegLossFactor) + '_rndShft_' + str(RandCropShift)
            torch.save(model.state_dict(), saveNamePrefix + '.pth')
            #torch.save(optimizer.state_dict(), saveNamePrefix + '.pth')

        # log
        logging.info('Epoch#%d: total loss=%e, Cls_loss=%e, Reg_loss=%e, Err_train=%e, Err_test=%e, lr = %f'
        %(epoch, train_loss_add, Cls_loss_add, Reg_loss_add, Error_train, Error_test, scheduler.get_lr()[0]))
        #logging.info('Epoch#%d: total loss=%e, Cls_loss=%e, Reg_loss=%e, Err_test=%e, lr = %f'
        #%(epoch, train_loss_add, Cls_loss_add, Reg_loss_add, Error_test, scheduler.get_lr()[0]))
        tbwriter.add_scalar('error', Error_test, epoch)
    tbwriter.close()
      
def test():   
    
    # net = model.Cls_Reg_Dual_Path_Net(num_classes = keypointsNumber)
    # net.load_state_dict(torch.load('/data/zhangboshen/CODE/Anchor_Pose_fpn/Results_icvl/176176_330K_tryscale/net_12_wetD_0.0001_depFact_0.5_RegFact_3_rndShft_5.pth')) 
    # net = net.cuda()
    # net.eval()
    model = get_model('test', keypointsNumber)
    model = DataParallel(model).cuda()
    model.load_state_dict(torch.load(model_dir))
    model.eval()


    # post_precess = anchor.post_process(shape=[cropHeight//16,cropWidth//16],stride=16,P_h=None, P_w=None)

    output = torch.FloatTensor()
    
    torch.cuda.synchronize() 
    time1 = time.time()
    
    # print('start calculating fps...')
    # fake_input = {
    #     'img':torch.ones((1,1,176,176),dtype=torch.float32).cuda(),
    #     'mask':torch.ones((1,176,176),dtype=torch.float32).cuda(),
    # }
    # fake_targets = {
    #     'joint_coord':torch.ones((1,21,3),dtype=torch.float32).cuda(),
    # }
    # fake_meta = {
    #     'joint_valid':torch.ones((1,21),dtype=torch.float32).cuda(),
    # }
    # t1 = time.time()
    # for i in range(0,10000):
    #     out = model(fake_input, fake_targets, fake_meta, 'test')
    # t2 = time.time()
    # print('time per img: ',(t2-t1)/10000)
    # print('fps: ',10000/(t2-t1))
    # input(":::")

    with torch.no_grad():
        for i, (inputs, targets, meta_info) in tqdm(enumerate(test_dataloaders)):    
        #torch.cuda.synchronize()

            # img, label = img.cuda(), label.cuda()
            inputs = {k:inputs[k].cuda() for k in inputs} 
            targets = {k:targets[k].cuda() for k in targets} 
            meta_info = {k:meta_info[k].cuda() for k in meta_info}
            out = model(inputs, targets, meta_info, 'test')
            pred_keypoints = out['joint_coord'] 
            
            # input_img = inputs['img']
            # input_mask = inputs['mask']
            # batch_size = input_img.shape[0]
            # for j in range(batch_size):
            #     img = input_img[j].cpu().numpy()
            #     img = img.transpose(1,2,0)  #   [C, H, W]--->>>[H, W, C] 
            #     img = np.ascontiguousarray(img)*255
            #     label = targets['joint_coord'][j]
            #     for i in label:
            #         cv2.circle(img, (int(i[0]),int(i[1])), 1, (0,255,255), 4)
            #     cv2.imwrite('img_inputs.jpg', img)
            #     input(':::')

            # pred_keypoints = post_precess(heads,voting=False)

            output = torch.cat([output,pred_keypoints.data.cpu()], 0)
            
            
    torch.cuda.synchronize()       
    time2 = time.time()
    print ('total_time',time2 - time1)
    print ('each_image_computing_time: ',(time2 - time1)/len(test_image_datasets))
    
    
    ## save the result for submition
    epoch = 0
    result = output.cpu().data.numpy()  # N, 21, 3
    writeTxt(result, center_test, epoch)
    print('saved the data for the result of epoch ', epoch)

        


    # save data
    # result = output.cpu().data.numpy()
    # #print('Error:', errorCompute(result,keypointsUVD_test, center_test, testingImageDir))
    # errTotal = errorCompute(result,keypointsUVD_test, center_test)
    # #errWrist, errTotal = errorCompute(result,keypointsUVD_train, center_train)
    # print('Error:', errTotal)
    

    # scio.savemat(os.path.join('/data/zhangboshen/CODE/Anchor_Pose_fpn/Results_icvl/176176_330K_tryscale/', '646_epoch12.mat'),{'result':result})
    #scio.savemat(os.path.join('/data/zhangboshen/CODE/Anchor_Pose_fpn/Results_nyu/xy_110_depth_150_160160_noncomp6', 'nyu_train_9.79_epoch18_UVD.mat'),{'UVD':UVD})


def errorCompute(source, target, center):
    assert np.shape(source)==np.shape(target), "source has different shape with target"

    Test1_ = source.copy()
    target_ = target.copy()
    # Test1_[:, :, 0] = source[:,:,1]
    # Test1_[:, :, 1] = source[:,:,0]
    Test1_[:, :, 0] = source[:,:,0]
    Test1_[:, :, 1] = source[:,:,1]
    Test1 = Test1_  # [x, y, z]
    
    center_pixel = center.copy()
    centre_world = pixel2world(center.copy(), 240.99, 240.96, 160, 120)

    centerlefttop = centre_world.copy()
    centerlefttop[:,0,0] = centerlefttop[:,0,0]-xy_thres
    centerlefttop[:,0,1] = centerlefttop[:,0,1]+xy_thres
    
    centerrightbottom = centre_world.copy()
    centerrightbottom[:,0,0] = centerrightbottom[:,0,0]+xy_thres
    centerrightbottom[:,0,1] = centerrightbottom[:,0,1]-xy_thres
    
    lefttop_pixel = world2pixel(centerlefttop, 240.99, 240.96, 160, 120)
    rightbottom_pixel = world2pixel(centerrightbottom, 240.99, 240.96, 160, 120)


    for i in range(len(Test1_)):
        Xmin = max(lefttop_pixel[i,0,0], 0)
        Ymin = max(rightbottom_pixel[i,0,1], 0)  
        Xmax = min(rightbottom_pixel[i,0,0], 160*2 - 1)
        Ymax = min(lefttop_pixel[i,0,1], 120*2 - 1)

        #Xmin = center[i][0][0] - cropHandWidth//2 
        #Ymin = center[i][0][1] - cropHandHeight//2 
        #Xmax = center[i][0][0] + cropHandWidth//2 
        #Ymax = center[i][0][1] + cropHandHeight//2 

        Test1[i,:,0] = Test1_[i,:,0]*(Xmax-Xmin)/cropWidth + Xmin  # x
        Test1[i,:,1] = Test1_[i,:,1]*(Ymax-Ymin)/cropHeight + Ymin  # y
        Test1[i,:,2] = source[i,:,2] + center[i][0][2]

    labels = pixel2world(target_, 240.99, 240.96, 160, 120)
    outputs = pixel2world(Test1.copy(), 240.99, 240.96, 160, 120)

    
    errors = np.sqrt(np.sum((labels - outputs) ** 2, axis=2))

    return np.mean(errors)
   

def writeTxt(result, center, epoch):

    resultUVD_ = result.copy()
    # resultUVD_[:, :, 0] = result[:,:,1]
    # resultUVD_[:, :, 1] = result[:,:,0]
    resultUVD_[:, :, 0] = result[:,:,0]
    resultUVD_[:, :, 1] = result[:,:,1]
    resultUVD = resultUVD_  # [x, y, z]
    
    center_pixel = center.copy()
    centre_world = pixel2world(center.copy(), 475.065948, 475.065857, 315.944855, 245.287079)

    centerlefttop = centre_world.copy()
    centerlefttop[:,0,0] = centerlefttop[:,0,0]-xy_thres
    centerlefttop[:,0,1] = centerlefttop[:,0,1]+xy_thres
    
    centerrightbottom = centre_world.copy()
    centerrightbottom[:,0,0] = centerrightbottom[:,0,0]+xy_thres
    centerrightbottom[:,0,1] = centerrightbottom[:,0,1]-xy_thres
    
    lefttop_pixel = world2pixel(centerlefttop, 475.065948, 475.065857, 315.944855, 245.287079)
    rightbottom_pixel = world2pixel(centerrightbottom, 475.065948, 475.065857, 315.944855, 245.287079)


    for i in range(len(result)):
        Xmin = max(lefttop_pixel[i,0,0], 0)
        Ymin = max(rightbottom_pixel[i,0,1], 0)  
        Xmax = min(rightbottom_pixel[i,0,0], 320*2 - 1)
        Ymax = min(lefttop_pixel[i,0,1], 240*2 - 1)

        resultUVD[i,:,0] = resultUVD_[i,:,0]*(Xmax-Xmin)/cropWidth + Xmin  # x
        resultUVD[i,:,1] = resultUVD_[i,:,1]*(Ymax-Ymin)/cropHeight + Ymin  # y
        resultUVD[i,:,2] = result[i,:,2] + center[i][0][2]

    resultXYD = pixel2world(resultUVD.copy(), 475.065948, 475.065857, 315.944855, 245.287079)

    resultReshape = resultXYD.reshape(len(resultXYD), -1)

    txtName = 'epoch_' + str(epoch) + '_XYD.txt'
    with open(os.path.join(save_dir, txtName), 'w') as f:     
        for i in tqdm(range(len(resultReshape))):
            f.write('frame/images/' + 'image_D%.8d'%(i+1) + '.png' + '\t')
            for j in range(keypointsNumber*3):
                f.write(str(resultReshape[i, j])+'\t')
            f.write('\n') 

    f.close()




if __name__ == '__main__':
    train()
    # test()