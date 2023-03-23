import numpy as np
import torch
import torch.utils.data
import cv2
import os.path as osp
from config import cfg
from utils.preprocessing import load_img, load_skeleton, get_bbox, process_bbox, augmentation, transform_input_to_output_space, trans_point2d
from utils.transforms import world2cam, cam2pixel, pixel2cam
from utils.vis import vis_keypoints, vis_3d_keypoints
import json
from pycocotools.coco import COCO
from tqdm import tqdm
import pickle


class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, mode):
        self.mode = mode # train, test, val
        self.img_path = cfg.interhand_images_path
        self.annot_path = cfg.interhand_anno_dir
        self.datalist_dir = cfg.datalistDir
        if self.mode == 'val':
            self.rootnet_output_path = '../rootnet_output/rootnet_interhand2.6m_output_val.json'
        else:
            self.rootnet_output_path = '../rootnet_output/rootnet_interhand2.6m_output_test.json'
        self.transform = transform
        self.joint_num = 21 # single hand
        self.root_joint_idx = {'right': 20, 'left': 41} 
        self.joint_type = {'right': np.arange(0,self.joint_num), 'left': np.arange(self.joint_num,self.joint_num*2)}
        self.skeleton = load_skeleton(osp.join(self.annot_path, 'skeleton.txt'), self.joint_num*2)
        self.use_single_hand_dataset = cfg.use_single_hand_dataset
        self.use_inter_hand_dataset = cfg.use_inter_hand_dataset
        self.vis = False
        
        ## use the total Interhand2.6M dataset
        datalist_file_path_sh = osp.join(self.datalist_dir , mode + '_datalist_sh_all.pkl')
        datalist_file_path_ih = osp.join(self.datalist_dir , mode + '_datalist_ih_all.pkl')
        
        # generate_new_datalist : whether to get datalist from existing file
        generate_new_datalist = True
        if osp.exists(datalist_file_path_sh) and osp.exists(datalist_file_path_ih):
            if (osp.getsize(datalist_file_path_sh) + osp.getsize(datalist_file_path_ih)) != 0:
                generate_new_datalist = False
        
        ## if the datalist is empty or doesn't exist, generate the pkl file and save the datalist
        if generate_new_datalist is True:
            self.datalist = []
            self.datalist_sh = []
            self.datalist_ih = []
            self.sequence_names = []
            
            # load annotation
            print("Load annotation from  " + osp.join(self.annot_path, self.mode))
            db = COCO(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_data.json'))
            with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_camera.json')) as f:
                cameras = json.load(f)
            with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_joint_3d.json')) as f:
                joints = json.load(f)

            # rootnet is not used
            if (self.mode == 'val' or self.mode == 'test') and cfg.trans_test == 'rootnet':
                print("Get bbox and root depth from " + self.rootnet_output_path)
                rootnet_result = {}
                with open(self.rootnet_output_path) as f:
                    annot = json.load(f)
                for i in range(len(annot)):
                    rootnet_result[str(annot[i]['annot_id'])] = annot[i]
            else:
                print("Get bbox and root depth from groundtruth annotation")

            # get images and annotations
            for aid in tqdm(list(db.anns.keys())[::1]):
                ann = db.anns[aid]
                image_id = ann['image_id']
                img = db.loadImgs(image_id)[0]
                hand_type = ann['hand_type']
                capture_id = img['capture']
                subject = img['subject']
                seq_name = img['seq_name']
                cam = img['camera']
                frame_idx = img['frame_idx'] 
                img_path = osp.join(self.img_path, self.mode, img['file_name'])

                campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
                focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
                joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)
                joint_cam = world2cam(joint_world.transpose(1,0), camrot, campos.reshape(3,1)).transpose(1,0)
                joint_img = cam2pixel(joint_cam, focal, princpt)[:,:2]
                joint_valid = np.array(ann['joint_valid'],dtype=np.float32).reshape(self.joint_num*2)

                ## Filter the data that does not meet the training requirements.
                ## All preprocessing refers to the baseline of Interhand2.6M(ECCV2020).
                # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
                joint_valid[self.joint_type['right']] *= joint_valid[self.root_joint_idx['right']]
                joint_valid[self.joint_type['left']] *= joint_valid[self.root_joint_idx['left']]
                # hand_type = ann['hand_type']
                hand_type_valid = np.array((ann['hand_type_valid']), dtype=np.float32)

                # rootnet is not used
                if (self.mode == 'val' or self.mode == 'test') and cfg.trans_test == 'rootnet':
                    bbox = np.array(rootnet_result[str(aid)]['bbox'],dtype=np.float32)
                    abs_depth = {'right': rootnet_result[str(aid)]['abs_depth'][0], 'left': rootnet_result[str(aid)]['abs_depth'][1]}
                else:
                    img_width, img_height = img['width'], img['height']
                    bbox = np.array(ann['bbox'],dtype=np.float32) # x,y,w,h
                    bbox = process_bbox(bbox, (img_height, img_width))
                    abs_depth = {'right': joint_cam[self.root_joint_idx['right'],2], 'left': joint_cam[self.root_joint_idx['left'],2]} #根节点的深度值，以此为参考
                
                cam_param = {'focal': focal, 'princpt': princpt}
                joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid}
                data = {'img_path': img_path, 'seq_name': seq_name, 'cam_param': cam_param, 
                        'bbox': bbox, 'joint': joint, 'hand_type': hand_type, 'hand_type_valid': hand_type_valid, 
                        'abs_depth': abs_depth, 'file_name': img['file_name'], 'capture': capture_id, 'cam': cam,
                        'frame': frame_idx, 'subject': subject, 'imgid': image_id
                }
                
                if hand_type == 'right' or hand_type == 'left':
                    if self.use_single_hand_dataset is True:
                        self.datalist_sh.append(data)
                elif hand_type == 'interacting':
                    if self.use_inter_hand_dataset is True:
                        self.datalist_ih.append(data)
                if seq_name not in self.sequence_names:
                    self.sequence_names.append(seq_name)
                    
            # Save the generated datalist to pkl file, easy to debug
            with open(datalist_file_path_sh, 'wb') as fs:
                pickle.dump(self.datalist_sh, fs)
            with open(datalist_file_path_ih, 'wb') as fi:
                pickle.dump(self.datalist_ih, fi)


        # Directly load the datalist saved in the previous file
        else:
            if self.use_single_hand_dataset is True:
                with open (datalist_file_path_sh, 'rb') as fsl:
                    self.datalist_sh = pickle.load(fsl)
            else:
                self.datalist_sh = []
            if self.use_inter_hand_dataset is True:
                with open (datalist_file_path_ih, 'rb') as fil:
                    self.datalist_ih = pickle.load(fil)
            else:
                self.datalist_ih = []

        self.datalist = self.datalist_sh + self.datalist_ih
        print('Number of annotations in single hand sequences: ' + str(len(self.datalist_sh)))
        print('Number of annotations in interacting hand sequences: ' + str(len(self.datalist_ih)))


    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1,0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0,1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1,1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)
    
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        data = self.datalist[idx]
        img_path, bbox, joint, hand_type, hand_type_valid = data['img_path'], data['bbox'], data['joint'], data['hand_type'], data['hand_type_valid']
        joint_cam = joint['cam_coord'].copy(); joint_img = joint['img_coord'].copy(); joint_valid = joint['valid'].copy();
        hand_type = self.handtype_str2array(hand_type)
        joint_coord = np.concatenate((joint_img, joint_cam[:,2,None].copy()),1)
        seq_name = data['seq_name']
        contact_vis_np = np.zeros((32, 2)).astype(np.float32)

        # image load
        img = load_img(img_path)

        # augmentation
        img, joint_coord, joint_valid, hand_type, inv_trans = augmentation(img, bbox, joint_coord, joint_valid, hand_type, self.mode, self.joint_type)
        rel_root_depth = np.array([joint_coord[self.root_joint_idx['left'],2] - joint_coord[self.root_joint_idx['right'],2]],dtype=np.float32).reshape(1)
        root_valid = np.array([joint_valid[self.root_joint_idx['right']] * joint_valid[self.root_joint_idx['left']]])*1.0

        # transform to output heatmap space
        joint_coord, joint_valid, rel_root_depth, root_valid =\
            transform_input_to_output_space(joint_coord, joint_valid, rel_root_depth, root_valid, self.root_joint_idx, self.joint_type)

        # Some images are blank, filter for training
        if np.sum(img) < 1e-4 :
            joint_valid *= 0
            root_valid *= 0
            hand_type_valid *= 0
            contact_vis_np *= 0

        img = self.transform(img.astype(np.float32)) / 255.

        # use zero mask.
        mask = np.zeros((img.shape[1], img.shape[2])).astype(np.bool)
        mask = self.transform(mask.astype(np.uint8))
        
        inputs = {'img': img, 'mask': mask}
        targets = {'joint_coord': joint_coord, 'rel_root_depth': rel_root_depth, 'hand_type': hand_type}
        meta_info = {'joint_valid': joint_valid, 'root_valid': root_valid, 'hand_type_valid': hand_type_valid, 
                     'inv_trans': inv_trans, 'capture': int(data['capture']), 'cam': int(data['cam']), 'frame': int(data['frame'])}
        return inputs, targets, meta_info


    def evaluate(self, preds):
        print() 
        print('Evaluation start...')

        gts = self.datalist
        preds_joint_coord, inv_trans, joint_valid_used = preds['joint_coord'], preds['inv_trans'], preds['joint_valid']
        assert len(gts) == len(preds_joint_coord)
        sample_num = len(gts)
        
        mpjpe_sh = [[] for _ in range(self.joint_num*2)]
        mpjpe_ih = [[] for _ in range(self.joint_num*2)]
        mpjpe_sh_2d = [[] for _ in range(self.joint_num*2)]
        mpjpe_sh_3d = [[] for _ in range(self.joint_num*2)]
        mpjpe_ih_2d = [[] for _ in range(self.joint_num*2)]
        mpjpe_ih_3d = [[] for _ in range(self.joint_num*2)]
        tot_err = []
        mpjpe_dict = {}


        mrrpe = []
        acc_hand_cls = 0; hand_cls_cnt = 0;
        for n in tqdm(range(sample_num),ncols=150):
            vis = False
            mpjpe_per_data_list = []
            mpjpe_per_data = 0

            data = gts[n]
            bbox, cam_param, joint, gt_hand_type, hand_type_valid = data['bbox'], data['cam_param'], data['joint'], data['hand_type'], data['hand_type_valid']
            hand_type = data['hand_type']

            focal = cam_param['focal']
            princpt = cam_param['princpt']
            gt_joint_coord = joint['cam_coord']  
            gt_joint_img = joint['img_coord']
            
            ## use original joint_valid param.
            joint_valid = joint['valid']
            # joint_valid = joint_valid_used[n]

            # restore xy coordinates to original image space
            pred_joint_coord_img = preds_joint_coord[n].copy()
            pred_joint_coord_img[:,0] = pred_joint_coord_img[:,0]/cfg.output_hm_shape[2]*cfg.input_img_shape[1]
            pred_joint_coord_img[:,1] = pred_joint_coord_img[:,1]/cfg.output_hm_shape[1]*cfg.input_img_shape[0]
            for j in range(self.joint_num*2):
                pred_joint_coord_img[j,:2] = trans_point2d(pred_joint_coord_img[j,:2],inv_trans[n])

            # restore depth to original camera space
            pred_joint_coord_img[:,2] = (pred_joint_coord_img[:,2]/cfg.output_hm_shape[0] * 2 - 1) * (cfg.bbox_3d_size/2)

            # add root joint depth
            pred_joint_coord_img[self.joint_type['right'],2] += data['abs_depth']['right']
            pred_joint_coord_img[self.joint_type['left'],2] += data['abs_depth']['left']

            # back project to camera coordinate system
            pred_joint_coord_cam = pixel2cam(pred_joint_coord_img, focal, princpt)

            # root joint alignment
            for h in ('right', 'left'):
                pred_joint_coord_cam[self.joint_type[h]] = pred_joint_coord_cam[self.joint_type[h]] - pred_joint_coord_cam[self.root_joint_idx[h],None,:]
                gt_joint_coord[self.joint_type[h]] = gt_joint_coord[self.joint_type[h]] - gt_joint_coord[self.root_joint_idx[h],None,:]
            

            # mpjpe
            ## xyz mpjpe
            for j in range(self.joint_num*2):
                if joint_valid[j]: ## 在这里，限制了只加载valid的坐标值
                    if gt_hand_type == 'right' or gt_hand_type == 'left':
                        mpjpe_sh[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))
                        mpjpe_per_data_list.append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))
                        # continue
                    else:
                        mpjpe_ih[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))
                        mpjpe_per_data_list.append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))
            

            ## xy mpjpe
            for j in range(self.joint_num*2):
                if joint_valid[j]:
                    if gt_hand_type == 'right' or gt_hand_type == 'left':
                        mpjpe_sh_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j,:2] - gt_joint_coord[j,:2])**2)))
                        # continue
                    else:
                        mpjpe_ih_2d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j,:2] - gt_joint_coord[j,:2])**2)))
            ## depth mpjpe
            for j in range(self.joint_num*2):
                if joint_valid[j]:
                    if gt_hand_type == 'right' or gt_hand_type == 'left':
                        mpjpe_sh_3d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j,2] - gt_joint_coord[j,2])**2)))
                        # continue
                    else:
                        mpjpe_ih_3d[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j,2] - gt_joint_coord[j,2])**2)))

            vis_2d = False    
            if vis_2d:
                img_path = data['img_path']
                cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                _img = cvimg[:,:,::-1].transpose(2,0,1)
                vis_kps = pred_joint_coord_img.copy()
                vis_kps_gt = gt_joint_img.copy()
                vis_valid = joint_valid.copy()
                capture = str(data['capture'])
                cam = str(data['cam'])
                frame = str(data['frame'])
                filename = 'out_' + str(n) + '_' + gt_hand_type + '.jpg'
                vis_keypoints(_img, vis_kps, vis_kps_gt, bbox, vis_valid, self.skeleton, filename)
                print('vis 2d over')

            
            vis_3d = False
            if vis_3d:
                filename = 'out_' + str(n) + '_3d.jpg'
                vis_3d_cam = pred_joint_coord_cam.copy()
                vis_3d_cam_left = pred_joint_coord_cam[self.joint_type['left']].copy()
                vis_3d_cam_left[:,2] = pred_joint_coord_cam[self.joint_type['left'],2]
                vis_3d_cam_right = pred_joint_coord_cam[self.joint_type['right']].copy()
                vis_3d_cam_right[:,2] = pred_joint_coord_cam[self.joint_type['right'],2] 
                vis_3d = np.concatenate((vis_3d_cam_left, vis_3d_cam_right), axis= 0)
                vis_3d_keypoints(vis_3d, joint_valid, self.skeleton, filename)
                print('vis 3d over')
                
        
        if hand_cls_cnt > 0: 
            handness_accuracy = acc_hand_cls / hand_cls_cnt
            print('Handedness accuracy: ' + str(handness_accuracy))
        if len(mrrpe) > 0: 
            mrrpe_num = sum(mrrpe)/len(mrrpe)
            print('MRRPE: ' + str(mrrpe_num))
        print()



        if self.use_inter_hand_dataset is True and self.use_single_hand_dataset is True:
            print('..................MPJPE FOR TOTAL HAND..................')
            eval_summary = 'MPJPE for each joint: \n'
            for j in range(self.joint_num*2):
                tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_sh[j]), np.stack(mpjpe_ih[j]))))
                joint_name = self.skeleton[j]['name']
                eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
                tot_err.append(tot_err_j)
            print(eval_summary)
            tot_err_mean = np.mean(tot_err)
            print('MPJPE for all hand sequences: %.2f' % (tot_err_mean))
            mpjpe_dict['total'] = tot_err_mean
            print()

        if self.use_single_hand_dataset is True:
            print('..................MPJPE FOR SINGLE HAND..................')
            ## xyz
            eval_summary = 'MPJPE for each joint: \n'
            for j in range(self.joint_num*2):
                mpjpe_sh[j] = np.mean(np.stack(mpjpe_sh[j]))
                joint_name = self.skeleton[j]['name']
                eval_summary += (joint_name + ': %.2f, ' % mpjpe_sh[j])
            print(eval_summary)
            mpjpe_sh_mean = np.mean(mpjpe_sh)
            print('MPJPE for single hand sequences: %.2f' % (mpjpe_sh_mean))
            mpjpe_dict['single_hand_total'] = mpjpe_sh_mean
            print()

            ## xy
            eval_summary_2d = 'MPJPE for each joint 2d: \n'
            for j in range(self.joint_num*2):
                mpjpe_sh_2d[j] = np.mean(np.stack(mpjpe_sh_2d[j]))
                joint_name = self.skeleton[j]['name']
                eval_summary_2d += (joint_name + ': %.2f, ' % mpjpe_sh_2d[j])
            print(eval_summary_2d) 
            mpjpe_sh_2d_mean = np.mean(mpjpe_sh_2d)
            print('MPJPE for single hand sequences 2d: %.2f' % (mpjpe_sh_2d_mean))
            mpjpe_dict['single_hand_2d'] = mpjpe_sh_2d_mean
            print()

            ## z
            eval_summary_3d = 'MPJPE for each joint depth: \n'
            for j in range(self.joint_num*2):
                mpjpe_sh_3d[j] = np.mean(np.stack(mpjpe_sh_3d[j]))
                joint_name = self.skeleton[j]['name']
                eval_summary_3d += (joint_name + ': %.2f, ' % mpjpe_sh_3d[j])
            print(eval_summary_3d) 
            mpjpe_sh_3d_mean = np.mean(mpjpe_sh_3d)
            print('MPJPE for single hand sequences 3d: %.2f' % (mpjpe_sh_3d_mean))
            mpjpe_dict['single_hand_depth'] = mpjpe_sh_3d_mean
            print()


        if self.use_inter_hand_dataset is True:
            print('..................MPJPE FOR INTER HAND..................')
            ## xyz
            eval_summary = 'MPJPE for each joint: \n'
            for j in range(self.joint_num*2):
                mpjpe_ih[j] = np.mean(np.stack(mpjpe_ih[j]))
                joint_name = self.skeleton[j]['name']
                eval_summary += (joint_name + ': %.2f, ' % mpjpe_ih[j])
            print(eval_summary) 
            mpjpe_ih_mean = np.mean(mpjpe_ih)
            print('MPJPE for interacting hand sequences: %.2f' % (mpjpe_ih_mean))
            mpjpe_dict['inter_hand_total'] = mpjpe_ih_mean
            print()

            ## xy
            eval_summary_2d = 'MPJPE for each joint 2d: \n'
            for j in range(self.joint_num*2):
                mpjpe_ih_2d[j] = np.mean(np.stack(mpjpe_ih_2d[j]))
                joint_name = self.skeleton[j]['name']
                eval_summary_2d += (joint_name + ': %.2f, ' % mpjpe_ih_2d[j])
            print(eval_summary_2d) 
            mpjpe_ih_2d_mean = np.mean(mpjpe_ih_2d)
            print('MPJPE for interacting hand sequences 2d: %.2f' % (mpjpe_ih_2d_mean))
            mpjpe_dict['inter_hand_2d'] = mpjpe_ih_2d_mean
            print()

            ## z
            eval_summary_3d = 'MPJPE for each joint depth: \n'
            for j in range(self.joint_num*2):
                mpjpe_ih_3d[j] = np.mean(np.stack(mpjpe_ih_3d[j]))
                joint_name = self.skeleton[j]['name']
                eval_summary_3d += (joint_name + ': %.2f, ' % mpjpe_ih_3d[j])
            print(eval_summary_3d) 
            mpjpe_ih_3d_mean = np.mean(mpjpe_ih_3d)
            print('MPJPE for interacting hand sequences 3d: %.2f' % (mpjpe_ih_3d_mean))
            mpjpe_dict['inter_hand_depth'] = mpjpe_ih_3d_mean
            print()


        if hand_cls_cnt > 0 and len(mrrpe) > 0:
            return mpjpe_dict, handness_accuracy, mrrpe_num
        else:
            return mpjpe_dict, None, None
        