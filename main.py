# -*- coding: utf-8 -*-
"""
Author: Ma Cheng Cheng
Description: 利用VGGnet的预训练模型来实现图像的检索，先用预训练模型来抽取图片的特征，然后把待检索的图像和数据库中的所有图像进行匹配，找出相似度最高的
"""

import streamlit as st
from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from keras.utils import image_utils
from numpy import linalg as LA
import h5py
import cv2
import numpy as np
import os

from extract_cnn_vgg16_keras import VGGNet


class VGGNet:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model_vgg = VGG16(weights=self.weight,
                               input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                               pooling=self.pooling, include_top=False)

    def vgg_extract_feat(self, img_path):
        img = image_utils.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image_utils.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input_vgg(img)
        feat = self.model_vgg.predict(img)
        norm_feat = feat[0] / LA.norm(feat[0])
        return norm_feat


def extract_feature(image_path, ):
    root = os.path.abspath('.')
    save_path = os.path.join(root, 'database', 'vgg_featureCNN.h5')
    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")
    imgpaths = []
    for subdir in os.listdir(image_path)[:]:
        curpath = os.path.join(image_path, subdir)
        imgpaths += [curpath]
    feats = []  # 保存图片特征向量
    model = VGGNet()
    for i, img_path in enumerate(imgpaths):
        norm_feat = model.vgg_extract_feat(img_path)
        feats.append(norm_feat)
        print("extracting feature from image No. %d , %d images in total" % ((i + 1), len(imgpaths)))
    feats = np.array(feats)
    print("--------------------------------------------------")
    print("      writing feature extraction results ...")
    print("--------------------------------------------------")
    h5f = h5py.File(save_path, 'w')
    h5f.create_dataset('dataset_1', data=feats)
    h5f.create_dataset('dataset_2', data=np.string_(imgpaths))
    h5f.close()
    print("             writing has ended.            ")
    h5f = h5py.File(save_path, 'r')


def find_img(maxres, query_imgname):
    root = os.path.abspath('.')
    save_path = os.path.join(root, 'database', 'vgg_featureCNN.h5')
    h5f = h5py.File(save_path, 'r')
    feats = h5f['dataset_1'][:]
    imgpaths = h5f['dataset_2'][:]
    h5f.close()
    # init VGGNet16 model
    model = VGGNet()
    # 待检索图片名
    print("--------------------------------------------------")
    print("               searching starts")
    print("--------------------------------------------------")

    # 提取待检索图片的特征
    queryVec = model.vgg_extract_feat(query_imgname)

    # 和数据库中的每张图片的特征匹配，计算匹配分数
    scores = np.dot(queryVec, feats.T)

    # 按匹配分数从大到小排序
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]
    imlist = []
    for i, index in enumerate(rank_ID[0:maxres]):
        imlist.append(imgpaths[index])
        print("image names: " + str(imgpaths[index]) + " scores: %f" % rank_score[i])
    print("top %d images in order are: " % maxres, imlist)

    # 输出检索到的图片
    for i, im in enumerate(imlist):
        impath = str(im)[2:-1]  # 得到的im是一个byte型的数据格式，需要转换成字符串
        image = cv2.imread(impath)
        # img = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = Image.open(image)
        st.image(image, caption="搜索输出 %d" % (i + 1), use_column_width='always')


def run():
    st.title('相似图片检索')
    image_path = st.sidebar.text_input('输入图片数据存储路径：')
    if image_path:
        extract_feature(image_path)
        preview = st.sidebar.file_uploader('选择一张待检测的图片', ['png', 'jpg', 'jpeg'])
        if preview:
            image = Image.open(preview)
            st.image(image, caption='待检测图片', use_column_width='always')
            option = st.slider('返回几张相似的图片?', 1, 10, 3)
            search_img = st.button(f"开始查找前{option}张相似图片")
            if search_img:
                find_img(option, preview)


if __name__ == '__main__':
    run()
