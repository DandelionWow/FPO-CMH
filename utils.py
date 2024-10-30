import os
import random
import numpy as np
import h5py
import importlib

from transformers import AutoTokenizer, AutoProcessor, AutoModel

import torch
import torch.nn.functional as F
import torch.utils.data as data

from dataset import BankInitDataset, OnlineLearningDataset

MODEL_CATEGORY = {
    'djsrh': 'ucmh',
    'jdsh': 'ucmh',
    'dgcpn': 'ucmh',
    'cirh': 'ucmh',
    'dcmh': 'scmh',
    'dadh': 'scmh',
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_data(root='', dataset_name='MSCOCO'):
    '''load data (feature and split_info)'''
    # feature
    feature_path = os.path.join(root, 'feature', '{}.mat'.format(dataset_name))
    feature = h5py.File(feature_path, 'r')
    # split index
    split_path = os.path.join(root, 'split', '{}_index.mat'.format(dataset_name))
    split = h5py.File(split_path, 'r')
    
    return feature, split


def load_dataloader(feature, split, mode='online_learning', phase='online', batch_size=1, shuffle=False):
    '''Load the dataloader based on features and split'''
    if mode == 'bank_init':
        idx = torch.Tensor(split[phase + '_train_index'][:]).type(torch.int64).squeeze()
        dataloader = data.DataLoader(dataset=BankInitDataset(feature['offline_image'][:].T[idx], feature['offline_text'][:].T[idx]), batch_size=batch_size, shuffle=shuffle)
    elif mode == 'online_learning':
        idx = torch.Tensor(split[phase + '_train_index'][:]).type(torch.int64).squeeze()
        dataloader = data.DataLoader(dataset=OnlineLearningDataset(feature['online_image'][:].T[idx], feature['online_text'][:].T[idx]), batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

def load_clip_model(name_or_path):
    '''load pretrained-clip model'''
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, local_files_only=True)
    processor = AutoProcessor.from_pretrained(name_or_path, local_files_only=True)
    clip_model = AutoModel.from_pretrained(name_or_path, local_files_only=True)
    clip_model = clip_model.cuda()
    return tokenizer, processor, clip_model


def load_ucmh(root='', model_name='cirh', code_len=16, dataset_name='MSCOCO'):
    '''load pretrained-ucmh model'''
    pretrained_model_path = os.path.join(root, 'pretrained_model', MODEL_CATEGORY[model_name.lower()], model_name.lower(), '{}_{}_{}bit_best_epoch.pth'.format(model_name.upper(), dataset_name, code_len))
    pretrained_model_ckpt = torch.load(pretrained_model_path)
    model = importlib.import_module(f'.{model_name.lower()}', package=f'models.{MODEL_CATEGORY[model_name.lower()]}')
    imgNet = getattr(model, 'ImgNet')(code_len=code_len)
    txtNet = getattr(model, 'TxtNet')(code_len=code_len)
    imgNet.load_state_dict(pretrained_model_ckpt['ImgNet'])
    txtNet.load_state_dict(pretrained_model_ckpt['TxtNet'])
    imgNet = imgNet.eval().cuda()
    txtNet = txtNet.eval().cuda()
    return imgNet, txtNet


def get_clip_category_feature(tokenizer, clip_model, template, clip_vocab_path):
    '''Get all categories and its feature according to the open vocabulary
    
    '''
    with open(clip_vocab_path, 'r') as f:
        categories = [s.strip('\n') for s in f.readlines()]
    
    if template == None or template == '':
        template = ['a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.', 'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.', 'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.', 'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.', 'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.', 'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']
    if isinstance(template, str):
        template = [template]
    
    categories_feature = []
    for clazz in categories:
        template_clazz = [t.format(clazz) for t in template]
        tokenized_template_clazz = tokenizer(template_clazz, padding=True, truncation=False, return_tensors="pt")
        tokenized_template_clazz['input_ids'] = tokenized_template_clazz['input_ids'].cuda()
        tokenized_template_clazz['attention_mask'] = tokenized_template_clazz['attention_mask'].cuda()
        with torch.no_grad():
            category_feature = clip_model.get_text_features(**tokenized_template_clazz)
        category_feature /= category_feature.norm(dim=-1, keepdim=True)
        category_feature = category_feature.mean(dim=0)
        category_feature /= category_feature.norm()
        categories_feature.append(category_feature)
    categories_feature = torch.stack(categories_feature)
    return categories, categories_feature


def zs_classifier(target: torch.Tensor, all_categories: torch.Tensor, threshold=0.1):
    '''clip zero-shot classifier

    Args:
        target: target feature\n
        all_categories: features of all categories\n
        threshold: minimum acceptance threshold 
    '''
    if len(target.size()) == 1:
        target = target.unsqueeze(0)
    target = target.cuda()
    all_categories = all_categories.cuda()

    similarity = (100. * F.normalize(target) @ F.normalize(all_categories).T).softmax(dim=-1)
    target_categories = torch.where(similarity[0] > threshold)[0].cpu().tolist()
    
    return target_categories

def compress(retrieval_loader, test_loader, model_I, model_T):
    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    with torch.no_grad():
        for _, (data_I, data_T, data_L, ) in enumerate(retrieval_loader):
            var_data_I = data_I.cuda()
            code_I = model_I(var_data_I)
            code_I = torch.sign(code_I)
            re_BI.extend(code_I)

            var_data_T = data_T.cuda()
            code_T = model_T(var_data_T)
            code_T = torch.sign(code_T)
            re_BT.extend(code_T)

            re_L.extend(data_L.cuda())

    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])

    with torch.no_grad():
        for _, (data_I, data_T, data_L, ) in enumerate(test_loader):
            var_data_I = data_I.cuda()
            code_I = model_I(var_data_I)
            code_I = torch.sign(code_I)
            qu_BI.extend(code_I)

            var_data_T = data_T.cuda()
            code_T = model_T(var_data_T)
            code_T = torch.sign(code_T)
            qu_BT.extend(code_T)

            qu_L.extend(data_L.cuda())

    re_BI = torch.stack(re_BI)
    re_BT = torch.stack(re_BT)
    re_L = torch.stack(re_L)

    qu_BI = torch.stack(qu_BI)
    qu_BT = torch.stack(qu_BT)
    qu_L = torch.stack(qu_L)

    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L


def calculate_top_map(qu_B, re_B, qu_L, re_L, topk=50):
    num_query = qu_L.shape[0]
    map = 0.
    if topk is None:
        topk = re_L.shape[0]

    for iter in range(num_query):
        q_L = qu_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)
        gnd = (q_L.mm(re_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        hamm = calc_hamming_dist(qu_B[iter, :], re_B)
        _, ind = torch.sort(hamm, stable=True)  # stable
        ind.squeeze_()
        gnd = gnd[ind]
        tgnd = gnd[:topk]
        tsum = torch.sum(tgnd)
        if tsum == 0:
            continue

        count = torch.arange(1, int(tsum) + 1).cuda().type(torch.float32)
        tindex = torch.nonzero(tgnd).squeeze().type(torch.float32) + 1.0
        map = map + torch.mean(count / tindex)
    
    map = map / num_query
    return map


def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH

