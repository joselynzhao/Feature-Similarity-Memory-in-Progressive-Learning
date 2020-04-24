import torch
from torch import nn
from reid import models
from reid.trainers import Trainer
from reid.evaluators import extract_features, Evaluator
from reid.dist_metric import DistanceMetric
import numpy as np
from collections import OrderedDict
import os.path as osp
import pickle
from reid.utils.serialization import load_checkpoint
from reid.utils.data import transforms as T
from torch.utils.data import DataLoader
from reid.utils.data.preprocessor import Preprocessor
import random
from sklearn.metrics.pairwise import cosine_similarity
# from run import outf
# import run
import  math
import codecs
import os


import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class REID():
    def __init__(self, model_name, batch_size, num_classes, data_dir, l_data, u_data, save_path, dropout=0.5, max_frames=100):

        self.model_name = model_name
        self.num_classes = num_classes
        self.data_dir = data_dir
        self.save_path = save_path

        self.l_data = l_data
        self.u_data = u_data
        self.l_label = np.array([label for _,label,_,_ in l_data])
        self.u_label = np.array([label for _,label,_,_ in u_data])

        self.dataloader_params = {}
        self.dataloader_params['height'] = 256
        self.dataloader_params['width'] = 128
        self.dataloader_params['batch_size'] = batch_size
        self.dataloader_params['workers'] = 6


        self.batch_size = batch_size
        self.data_height = 256
        self.data_width = 128
        self.data_workers = 6

        # batch size for eval mode. Default is 1.
        self.eval_bs = 1
        self.dropout = dropout
        self.max_frames = max_frames


    def get_dataloader(self, dataset, training=False) :
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        if training:
            transformer = T.Compose([
                T.RandomSizedRectCrop(self.data_height, self.data_width),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalizer,
            ])
            batch_size = self.batch_size

        else:
            transformer = T.Compose([
                T.RectScale(self.data_height, self.data_width),
                T.ToTensor(),
                normalizer,
            ])
            batch_size = self.eval_bs

        data_loader = DataLoader(
            Preprocessor(dataset, root=self.data_dir,
                         transform=transformer, is_training=training, max_frames=self.max_frames),
            batch_size=batch_size, num_workers=self.data_workers,
            shuffle=training, pin_memory=True, drop_last=training)

        current_status = "Training" if training else "Test"
        print("create dataloader for {} with batch_size {}".format(current_status, batch_size))
        return data_loader

    def train(self, train_data, step, epochs=10, step_size=55, init_lr=0.1, dropout=0.5):

        """ create model and dataloader """
        model = models.create(self.model_name, dropout=self.dropout, num_classes=self.num_classes, mode=self.mode)
        model = nn.DataParallel(model).cuda()
        # model = nn.DataParallel(model, device_ids=[3,4]).cuda()
        # model.to(device)
        dataloader = self.get_dataloader(train_data, training=True)

        # the base parameters for the backbone (e.g. ResNet50)
        base_param_ids = set(map(id, model.module.CNN.base.parameters()))

        # we fixed the first three blocks to save GPU memory
        base_params_need_for_grad = filter(lambda p: p.requires_grad, model.module.CNN.parameters())

        # params of the new layers
        new_params = [p for p in model.parameters() if id(p) not in base_param_ids]

        # set the learning rate for backbone to be 0.1 times
        param_groups = [
            {'params': base_params_need_for_grad, 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]

        criterion = nn.CrossEntropyLoss().cuda()  # 标准
        optimizer = torch.optim.SGD(param_groups, lr=init_lr, momentum=0.5, weight_decay = 5e-4, nesterov=True)

        # change the learning rate by step
        def adjust_lr(epoch, step_size):     #学习率的衰减也可以做调整
            lr = init_lr / (10 ** (epoch // step_size))
            for g in optimizer.param_groups:
                g['lr'] = lr * g.get('lr_mult', 1)

            if epoch % step_size == 0:
                print("Epoch {}, current lr {}".format(epoch, lr))

        """ main training process """
        trainer = Trainer(model, criterion)
        for epoch in range(epochs):
            adjust_lr(epoch, step_size)
            trainer.train(epoch, dataloader, optimizer)
            # trainer.train(epoch, dataloader, optimizer, print_freq=len(dataloader)//30 * 10)
        # if tagper == 1:
        #     save_path = osp.join(self.save_path,'tagper1')
        #     if os.path.exists(save_path) is False:
        #         os.makedirs(save_path)
        # elif tagper == 2:
        #     save_path = osp.join(self.save_path,'tagper2')
        #     if os.path.exists(save_path) is False:
        #         os.makedirs(save_path)
        save_path = self.save_path
        torch.save(model.state_dict(), osp.join(save_path, "{}_step_{}.ckpt".format(self.mode, step)))
        self.model = model


    def get_feature(self, dataset):
        dataloader = self.get_dataloader(dataset, training=False)
        features,_ = extract_features(self.model, dataloader)
        features = np.array([logit.numpy() for logit in features.values()])
        return features

    def estimate_label(self):
        # extract feature
        u_feas = self.get_feature(self.u_data)
        l_feas = self.get_feature(self.l_data)
        print("u_features", u_feas.shape, "l_features", l_feas.shape)
        scores = np.zeros((u_feas.shape[0]))
        labels = np.zeros((u_feas.shape[0]))
        # 分别用来存 _ufeas的分数和标签
        id_num = {}  # 以标签名称作为字典
        # a = 1
        num_correct_pred = 0
        for idx, u_fea in enumerate(u_feas):
            diffs = l_feas - u_fea
            dist = np.linalg.norm(diffs, axis=1)
            index_min = np.argmin(dist)
            scores[idx] = - dist[index_min]  # "- dist" : more dist means less score
            labels[idx] = self.l_label[index_min]  # take the nearest labled neighbor as the prediction label
            if self.u_label[idx] == labels[idx]:
                num_correct_pred += 1
        label_pre = num_correct_pred / u_feas.shape[0]
        print("predictions on all the unlabeled data: {} of {} is correct, accuracy = {:0.3f}".format(
            num_correct_pred, u_feas.shape[0], label_pre))
        return labels, scores, label_pre




    def select_top_data(self, pred_score, nums_to_select):
        v = np.zeros(len(pred_score))
        index = np.argsort(-pred_score)
        for i in range(nums_to_select):  #排序,求最前面的n个
            v[index[i]] = 1
        return v.astype('bool')


    def generate_new_train_data(self, sel_idx, pred_y,u_data):
        """ generate the next training data """
        u_label =  np.array([label for _,label,_,_ in u_data])
        seletcted_data = []
        correct, total = 0, 0
        for i, flag in enumerate(sel_idx):
            if flag: # if selected
                seletcted_data.append([u_data[i][0], int(pred_y[i]), u_data[i][2], u_data[i][3]])
                total += 1
                if u_label[i] == int(pred_y[i]):
                    correct += 1
        if total == 0:
            acc = 1
        else : acc = correct / total

        new_train_data = seletcted_data
        print("selected pseudo-labeled data: {} of {} is correct, accuracy: {:0.4f}  new train data: {}".format(
                correct, len(seletcted_data), acc, len(new_train_data)))

        return new_train_data,acc


    def generate_new_train_data_only(self,sel_idx,pred_y,u_data):
        seletcted_data = []
        for i,flag in enumerate(sel_idx):
            if flag:
                seletcted_data.append([u_data[i][0], int(pred_y[i]), u_data[i][2], u_data[i][3]])
        return seletcted_data

    def get_select_pre(self,sel_idx,pred_y,u_data):
        u_label = np.array([label for _, label, _, _ in u_data])
        correct,total =0,0
        for i,flag in enumerate(sel_idx):
            if flag:
                total +=1
                if u_label[i] == int(pred_y[i]):
                    correct +=1
        if total ==0:
            acc = 0
        else: acc = correct /total
        return acc

    def resume(self, ckpt_file, step):
        print("continued from step", step)
        model = models.create(self.model_name, dropout=self.dropout, num_classes=self.num_classes, mode=self.mode)
        self.model = nn.DataParallel(model).cuda()
        self.model.load_state_dict(load_checkpoint(ckpt_file))

    def evaluate(self, query, gallery):
        test_loader = self.get_dataloader(list(set(query) | set(gallery)), training = False)
        evaluator = Evaluator(self.model)
        return  evaluator.evaluate(test_loader, query, gallery)



"""
    Get one-shot split for the input dataset.
"""
def get_one_shot_in_cam1(dataset, load_path, seed=0):

    np.random.seed(seed)
    random.seed(seed)

    # if previous split exists, load it and return
    if osp.exists(load_path):
        with open(load_path, "rb") as fp:
            dataset = pickle.load(fp)
            label_dataset = dataset["label set"]
            unlabel_dataset = dataset["unlabel set"]

        print("  labeled  |   N/A | {:8d}".format(len(label_dataset)))
        print("  unlabel  |   N/A | {:8d}".format(len(unlabel_dataset)))
        print("\nLoad one-shot split from", load_path)
        return label_dataset, unlabel_dataset



    #print("random create new one-shot split and save it to", load_path)

    label_dataset = []
    unlabel_dataset = []

    # dataset indexed by [pid, cam]
    dataset_in_pid_cam = [[[] for _ in range(dataset.num_cams)] for _ in range(dataset.num_train_ids) ]
    for index, (images, pid, camid, videoid) in enumerate(dataset.train):
        dataset_in_pid_cam[pid][camid].append([images, pid, camid, videoid])


    # generate the labeled dataset by randomly selecting a tracklet from the first camera for each identity
    for pid, cams_data  in enumerate(dataset_in_pid_cam):
        for camid, videos in enumerate(cams_data):
            if len(videos) != 0:
                selected_video = random.choice(videos)
                break
        label_dataset.append(selected_video)
    assert len(label_dataset) == dataset.num_train_ids
    labeled_videoIDs =[vid for _, (_,_,_, vid) in enumerate(label_dataset)]

    # generate unlabeled set
    for (imgs, pid, camid, videoid) in dataset.train:
        if videoid not in labeled_videoIDs:
            unlabel_dataset.append([imgs, pid, camid, videoid])


    with open(load_path, "wb") as fp:
        pickle.dump({"label set":label_dataset, "unlabel set":unlabel_dataset}, fp)


    print("  labeled    | N/A | {:8d}".format(len(label_dataset)))
    print("  unlabeled  | N/A | {:8d}".format(len(unlabel_dataset)))
    print("\nCreate new one-shot split, and save it to", load_path)
    return label_dataset, unlabel_dataset

