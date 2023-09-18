from curses import meta
from re import I
import cv2
import torch
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import scipy.io as scio
import os
from PIL import Image
from torch.autograd import Variable
# import model as model
# import anchor as anchor
from tqdm import tqdm
import random_erasing
import logging
import time
import datetime
import random

from model import get_model
from torch.nn.parallel.data_parallel import DataParallel

from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

fx = 588.03
fy = -587.07
u0 = 320
v0 = 240

# DataHyperParms 
TrainImgFrames = 72757
TestImgFrames = 8252
keypointsNumber = 14
cropWidth = 176
cropHeight = 176
batch_size = 4

lr = 1e-4
lr_dec_epoch = [20,25,30,35,40]
lr_dec_factor = 5

Weight_Decay = 1e-4
nepoch = 45
RegLossFactor = 3
spatialFactor = 0.5
RandCropShift = 5
RandshiftDepth = 1
RandRotate = 180 
RandScale = (1.0, 0.5)
xy_thres = 110
depth_thres = 150

randomseed = 12345
random.seed(randomseed)
np.random.seed(randomseed)
torch.manual_seed(randomseed)

save_dir = '/data/data2/jiangchanglong/a2jformer/base/output/model_dump_new_depth'
tensorboard_dir = '/data/data2/jiangchanglong/a2jformer/base/output/tensorboard_log'
'''
commend for tensorboard is : 
tensorboard --logdir=/data/data2/jiangchanglong/a2jformer/base/output/tensorboard_log
'''

try:
    os.makedirs(save_dir)
except OSError:
    pass


trainingImageDir = '/data/data1/zhangboshen/CODE/219_A2J_original/Anchor_Pose_fpn/data/nyu/train_nyu/'
testingImageDir = '/data/data1/zhangboshen/CODE/219_A2J_original/Anchor_Pose_fpn/data/nyu/test_nyu/'  # mat images
test_center_file = '/data/data2/jiangchanglong/a2jformer/code3xyz/data/nyu/nyu_center_test.mat'
test_keypoint_file = '/data/data2/jiangchanglong/a2jformer/code3xyz/data/nyu/nyu_keypointsUVD_test.mat'
train_center_file = '/data/data2/jiangchanglong/a2jformer/code3xyz/data/nyu/nyu_center_train.mat'
train_keypoint_file = '/data/data2/jiangchanglong/a2jformer/code3xyz/data/nyu/nyu_keypointsUVD_train.mat'
MEAN = np.load('/data/data2/jiangchanglong/a2jformer/code3xyz/data/nyu/nyu_mean.npy')
STD = np.load('/data/data2/jiangchanglong/a2jformer/code3xyz/data/nyu/nyu_std.npy')
model_dir = '/data/data2/jiangchanglong/a2jformer/base/output/model_dump_new_depth/model_41_wetD_0.0001_depFact_0.5_RegFact_3_rndShft_5.pth'
result_file = '/data/data2/jiangchanglong/a2jformer/code3xyz/output_nyu/result_NYU.txt'


def pixel2world(x, fx, fy, ux, uy):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x


def world2pixel(x, fx, fy, ux, uy):
    x[:, :, 0] = x[:, :, 0] * fx / x[:, :, 2] + ux
    x[:, :, 1] = x[:, :, 1] * fy / x[:, :, 2] + uy
    return x
    
joint_id_to_name = {
  0: 'pinky tip',
  1: 'pinky mid',
  2: 'ring tip',
  3: 'ring mid',
  4: 'middle tip',
  5: 'middle mid',
  6: 'index tip',
  7: 'index mid',
  8: 'thumb tip',
  9: 'thumb mid',
  10: 'thumb root',
  11: 'wrist back',
  12: 'wrist',
  13: 'palm',
}

## loading GT keypoints and center points
keypointsUVD_test = scio.loadmat(test_keypoint_file)['keypoints3D'].astype(np.float32)      
center_test = scio.loadmat(test_center_file)['centre_pixel'].astype(np.float32)

centre_test_world = pixel2world(center_test.copy(), fx, fy, u0, v0)

centerlefttop_test = centre_test_world.copy()
centerlefttop_test[:,0,0] = centerlefttop_test[:,0,0]-xy_thres
centerlefttop_test[:,0,1] = centerlefttop_test[:,0,1]+xy_thres

centerrightbottom_test = centre_test_world.copy()
centerrightbottom_test[:,0,0] = centerrightbottom_test[:,0,0]+xy_thres
centerrightbottom_test[:,0,1] = centerrightbottom_test[:,0,1]-xy_thres

test_lefttop_pixel = world2pixel(centerlefttop_test, fx, fy, u0, v0)
test_rightbottom_pixel = world2pixel(centerrightbottom_test, fx, fy, u0, v0)


keypointsUVD_train = scio.loadmat(train_keypoint_file)['keypoints3D'].astype(np.float32)      
center_train = scio.loadmat(train_center_file)['centre_pixel'].astype(np.float32)
centre_train_world = pixel2world(center_train.copy(), fx, fy, u0, v0)

centerlefttop_train = centre_train_world.copy()
centerlefttop_train[:,0,0] = centerlefttop_train[:,0,0]-xy_thres
centerlefttop_train[:,0,1] = centerlefttop_train[:,0,1]+xy_thres

centerrightbottom_train = centre_train_world.copy()
centerrightbottom_train[:,0,0] = centerrightbottom_train[:,0,0]+xy_thres
centerrightbottom_train[:,0,1] = centerrightbottom_train[:,0,1]-xy_thres

train_lefttop_pixel = world2pixel(centerlefttop_train, fx, fy, u0, v0)
train_rightbottom_pixel = world2pixel(centerrightbottom_train, fx, fy, u0, v0)

def transform(img, label, matrix):
    '''
    img: [H, W]  label, [N,2]   
    '''
    img_out = cv2.warpAffine(img,matrix,(cropWidth,cropHeight))
    label_out = np.ones((keypointsNumber, 3))
    label_out[:,:2] = label[:,:2].copy()
    label_out = np.matmul(matrix, label_out.transpose())
    label_out = label_out.transpose()

    return img_out, label_out

def dataPreprocess(index, img, keypointsUVD, center, mean, std, lefttop_pixel, rightbottom_pixel, xy_thres=90, depth_thres=75, augment=True):
    
    imageOutputs = np.ones((cropHeight, cropWidth, 1), dtype='float32') 
    labelOutputs = np.ones((keypointsNumber, 3), dtype = 'float32') 
    
    if augment:
        RandomOffset_1 = np.random.randint(-1*RandCropShift,RandCropShift)
        RandomOffset_2 = np.random.randint(-1*RandCropShift,RandCropShift)
        RandomOffset_3 = np.random.randint(-1*RandCropShift,RandCropShift)
        RandomOffset_4 = np.random.randint(-1*RandCropShift,RandCropShift)
        RandomOffsetDepth = np.random.normal(0, RandshiftDepth, cropHeight*cropWidth).reshape(cropHeight,cropWidth) 
        RandomOffsetDepth[np.where(RandomOffsetDepth < RandshiftDepth)] = 0
        RandomRotate = np.random.randint(-1*RandRotate,RandRotate)
        RandomScale = np.random.rand()*RandScale[0]+RandScale[1]
        matrix = cv2.getRotationMatrix2D((cropWidth/2,cropHeight/2),RandomRotate,RandomScale)
    else:
        RandomOffset_1, RandomOffset_2, RandomOffset_3, RandomOffset_4 = 0, 0, 0, 0
        RandomRotate = 0
        RandomScale = 1
        RandomOffsetDepth = 0
        matrix = cv2.getRotationMatrix2D((cropWidth/2,cropHeight/2),RandomRotate,RandomScale)

    new_Xmin = max(lefttop_pixel[index,0,0] + RandomOffset_1, 0)
    new_Ymin = max(lefttop_pixel[index,0,1] + RandomOffset_2, 0)
    new_Xmax = min(rightbottom_pixel[index,0,0] + RandomOffset_3, img.shape[1] - 1)
    new_Ymax = min(rightbottom_pixel[index,0,1] + RandomOffset_4, img.shape[0] - 1)

    imCrop = img[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)].copy()

    imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)

    imgResize = np.asarray(imgResize,dtype = 'float32')  # H*W*C

    imgResize[np.where(imgResize >= center[index][0][2] + depth_thres)] = center[index][0][2] 
    imgResize[np.where(imgResize <= center[index][0][2] - depth_thres)] = center[index][0][2] 
    imgResize = (imgResize - center[index][0][2])*RandomScale

    imgResize = (imgResize - mean) / std

    ## label
    label_xy = np.ones((keypointsNumber, 2), dtype = 'float32') 
    label_xy[:,0] = (keypointsUVD[index,:,0].copy() - new_Xmin)*cropWidth/(new_Xmax - new_Xmin) # x
    label_xy[:,1] = (keypointsUVD[index,:,1].copy() - new_Ymin)*cropHeight/(new_Ymax - new_Ymin) # y

    if augment:
        imgResize, label_xy = transform(imgResize, label_xy, matrix)  ## rotation, scale
    
    imageOutputs[:,:,0] = imgResize

    # labelOutputs[:,1] = label_xy[:,0]
    # labelOutputs[:,0] = label_xy[:,1]
    labelOutputs[:,1] = label_xy[:,1]
    labelOutputs[:,0] = label_xy[:,0]


    labelOutputs[:,2] = (keypointsUVD[index,:,2] - center[index][0][2])*RandomScale   # Z  
    
    imageOutputs = np.asarray(imageOutputs)
    imageNCHWOut = imageOutputs.transpose(2, 0, 1)  # [H, W, C] --->>>  [C, H, W]
    imageNCHWOut = np.asarray(imageNCHWOut)
    labelOutputs = np.asarray(labelOutputs)

    data, label = torch.from_numpy(imageNCHWOut), torch.from_numpy(labelOutputs)

    return data, label


######################   Pytorch dataloader   #################
class my_dataloader(torch.utils.data.Dataset):

    def __init__(self, ImgDir, center, lefttop_pixel, rightbottom_pixel, keypointsUVD, augment=True):

        self.ImgDir = ImgDir
        self.mean = MEAN
        self.std = STD
        self.center = center
        self.lefttop_pixel = lefttop_pixel
        self.rightbottom_pixel = rightbottom_pixel
        self.keypointsUVD = keypointsUVD
        self.xy_thres = xy_thres
        self.depth_thres = depth_thres
        self.augment = augment
        self.bbox_3d_size = 400
        self.output_hm_shape = 64
        self.randomErase = random_erasing.RandomErasing(probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0])

    def __getitem__(self, index):

        depth = scio.loadmat(self.ImgDir + str(index+1) + '.mat')['depth']       

        data, label = dataPreprocess(index, depth, self.keypointsUVD, self.center, self.mean, self.std, \
            self.lefttop_pixel, self.rightbottom_pixel, self.xy_thres, self.depth_thres, self.augment)

        if self.augment:
            data = self.randomErase(data)

        # use zero mask for now. Later if required put ones along padded pixels
        mask = np.zeros((data.shape[1], data.shape[2])).astype(np.bool)
        # print(torch.max(data))
        # print(torch.min(data))
        

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

      
train_image_datasets = my_dataloader(trainingImageDir, center_train, train_lefttop_pixel, train_rightbottom_pixel, keypointsUVD_train, augment=True)
train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size = batch_size,
                                             shuffle = True, num_workers = 8)

test_image_datasets = my_dataloader(testingImageDir, center_test, test_lefttop_pixel, test_rightbottom_pixel, keypointsUVD_test, augment=False)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size = batch_size,
                                             shuffle = False, num_workers = 8)

def train():
    model = get_model('train', keypointsNumber)
    # net = model.A2J_model(num_classes = keypointsNumber)
    # net = net.cuda()
    
    model = DataParallel(model).cuda()
    tbwriter = SummaryWriter(tensorboard_dir)
    
    # post_precess = anchor.post_process(shape=[cropHeight//16,cropWidth//16],stride=16,P_h=None, P_w=None)
    # criterion = anchor.A2J_loss(shape=[cropHeight//16,cropWidth//16],thres = [16.0,32.0],stride=16,\
    #     spatialFactor=spatialFactor,img_shape=[cropHeight, cropWidth],P_h=None, P_w=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=Weight_Decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)
    
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                    filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
    logging.info('======================================================')

    for epoch in range(nepoch):
        model = model.train()
        train_loss_add = 0.0
        Cls_loss_add = 0.0
        Reg_loss_add = 0.0
        
        timer = time.time()
    
        # Training loop
        for i, (inputs, targets, meta_info) in enumerate(train_dataloaders):

            torch.cuda.synchronize() 
            # img, label = img.cuda(), label.cuda()   
            inputs = {k:inputs[k].cuda() for k in inputs} 
            targets = {k:targets[k].cuda() for k in targets} 
            meta_info = {k:meta_info[k].cuda() for k in meta_info} 

            loss = model(inputs, targets, meta_info, 'train')
            loss = {k:loss[k].mean() for k in loss}
            
            Cls_loss, Reg_loss, total_loss = loss['Cls_loss'], loss['Reg_loss'], loss['total_loss']

            # heads  = net(img)  
            #print(regression)     
            optimizer.zero_grad()  
            # Cls_loss, Reg_loss = criterion(heads, label)
            # loss = 1*Cls_loss + Reg_loss*RegLossFactor
            # loss.backward()
            loss['total_loss'].backward()

            optimizer.step()
            torch.cuda.synchronize()
            
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
        Error_test_wrist = 0

        if (epoch % 1 == 0):  
            model = model.eval()
            output = torch.FloatTensor()
            outputTrain = torch.FloatTensor()

            with torch.no_grad():
                for i, (inputs, targets, meta_info) in tqdm(enumerate(test_dataloaders)):
                    inputs = {k:inputs[k].cuda() for k in inputs} 
                    targets = {k:targets[k].cuda() for k in targets} 
                    meta_info = {k:meta_info[k].cuda() for k in meta_info} 
                    # img, label = img.cuda(), label.cuda()       
                    # heads = model(img)
                    out = model(inputs, targets, meta_info, 'test')
                    pred_keypoints = out['joint_coord']
                    # pred_keypoints = post_precess(heads, voting=False)
                    output = torch.cat([output,pred_keypoints.data.cpu()], 0)

            result = output.cpu().data.numpy()
            Error_test = errorCompute(result, keypointsUVD_test, center_test)
            print('epoch: ', epoch, 'Test error:', Error_test)
            saveNamePrefix = '%s/model_%d_wetD_' % (save_dir, epoch) + str(Weight_Decay) + '_depFact_' + str(spatialFactor) + '_RegFact_' + str(RegLossFactor) + '_rndShft_' + str(RandCropShift)
            torch.save(model.state_dict(), saveNamePrefix + '.pth')

        # log
        logging.info('Epoch#%d: total loss=%.4f, Cls_loss=%.4f, Reg_loss=%.4f, Err_test=%.4f, lr = %.6f'
        %(epoch, train_loss_add, Cls_loss_add, Reg_loss_add, Error_test, scheduler.get_lr()[0]))
        tbwriter.add_scalar('error', Error_test, epoch)
    tbwriter.close()


def test():   
    # net = model.A2J_model(num_classes = keypointsNumber)
    model = get_model('test', keypointsNumber)
    model = DataParallel(model).cuda()

    model.load_state_dict(torch.load(model_dir))
    # model = model.cuda()
    model.eval()
    
    # post_precess = anchor.post_process(shape=[cropHeight//16,cropWidth//16],stride=16,P_h=None, P_w=None)

    output = torch.FloatTensor()
    torch.cuda.synchronize() 
    
            
    # print('start calculating fps...')
    # fake_input = {
    #     'img':torch.ones((1,1,176,176),dtype=torch.float32).cuda(),
    #     'mask':torch.ones((1,176,176),dtype=torch.float32).cuda(),
    # }
    # fake_targets = {
    #     'joint_coord':torch.ones((1,14,3),dtype=torch.float32).cuda(),
    # }
    # fake_meta = {
    #     'joint_valid':torch.ones((1,14),dtype=torch.float32).cuda(),
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
            # img, label = img.cuda(), label.cuda()    
            inputs = {k:inputs[k].cuda() for k in inputs} 
            targets = {k:targets[k].cuda() for k in targets} 
            meta_info = {k:meta_info[k].cuda() for k in meta_info}
            out = model(inputs, targets, meta_info, 'test')
            pred_keypoints = out['joint_coord'] 
            # heads = model(img)  
            # pred_keypoints = post_precess(heads,voting=False)
            output = torch.cat([output,pred_keypoints.data.cpu()], 0)
        
    torch.cuda.synchronize()       

    result = output.cpu().data.numpy()
    writeTxt(result, center_test)
    error = errorCompute(result, keypointsUVD_test, center_test)
    print('Error:', error)
    

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
    centre_world = pixel2world(center.copy(), fx, fy, u0, v0)

    centerlefttop = centre_world.copy()
    centerlefttop[:,0,0] = centerlefttop[:,0,0]-xy_thres
    centerlefttop[:,0,1] = centerlefttop[:,0,1]+xy_thres
    
    centerrightbottom = centre_world.copy()
    centerrightbottom[:,0,0] = centerrightbottom[:,0,0]+xy_thres
    centerrightbottom[:,0,1] = centerrightbottom[:,0,1]-xy_thres
    
    lefttop_pixel = world2pixel(centerlefttop, fx, fy, u0, v0)
    rightbottom_pixel = world2pixel(centerrightbottom, fx, fy, u0, v0)

    for i in range(len(Test1_)):
        Xmin = max(lefttop_pixel[i,0,0], 0)
        Ymin = max(lefttop_pixel[i,0,1], 0)
        Xmax = min(rightbottom_pixel[i,0,0], 320*2 - 1)
        Ymax = min(rightbottom_pixel[i,0,1], 240*2 - 1)

        Test1[i,:,0] = Test1_[i,:,0]*(Xmax-Xmin)/cropWidth + Xmin  # x
        Test1[i,:,1] = Test1_[i,:,1]*(Ymax-Ymin)/cropHeight + Ymin  # y
        Test1[i,:,2] = source[i,:,2] + center[i][0][2]

    labels = pixel2world(target_, fx, fy, u0, v0)
    outputs = pixel2world(Test1.copy(), fx, fy, u0, v0)

    errors = np.sqrt(np.sum((labels - outputs) ** 2, axis=2))

    return np.mean(errors)
   

   
def writeTxt(result, center):

    resultUVD_ = result.copy()
    resultUVD_[:, :, 0] = result[:,:,1]
    resultUVD_[:, :, 1] = result[:,:,0]
    resultUVD = resultUVD_  # [x, y, z]
    
    center_pixel = center.copy()
    centre_world = pixel2world(center.copy(), fx, fy, u0, v0)

    centerlefttop = centre_world.copy()
    centerlefttop[:,0,0] = centerlefttop[:,0,0]-xy_thres
    centerlefttop[:,0,1] = centerlefttop[:,0,1]+xy_thres
    
    centerrightbottom = centre_world.copy()
    centerrightbottom[:,0,0] = centerrightbottom[:,0,0]+xy_thres
    centerrightbottom[:,0,1] = centerrightbottom[:,0,1]-xy_thres
    
    lefttop_pixel = world2pixel(centerlefttop, fx, fy, u0, v0)
    rightbottom_pixel = world2pixel(centerrightbottom, fx, fy, u0, v0)


    for i in range(len(result)):
        Xmin = max(lefttop_pixel[i,0,0], 0)
        Ymin = max(lefttop_pixel[i,0,1], 0)
        Xmax = min(rightbottom_pixel[i,0,0], 320*2 - 1)
        Ymax = min(rightbottom_pixel[i,0,1], 240*2 - 1)

        resultUVD[i,:,0] = resultUVD_[i,:,0]*(Xmax-Xmin)/cropWidth + Xmin  # x
        resultUVD[i,:,1] = resultUVD_[i,:,1]*(Ymax-Ymin)/cropHeight + Ymin  # y
        resultUVD[i,:,2] = result[i,:,2] + center[i][0][2]

    resultReshape = resultUVD.reshape(len(result), -1)

    with open(os.path.join(save_dir, result_file), 'w') as f:     
        for i in range(len(resultReshape)):
            for j in range(keypointsNumber*3):
                f.write(str(resultReshape[i, j])+' ')
            f.write('\n') 

    f.close()

if __name__ == '__main__':
    train()
    test()

    
      
