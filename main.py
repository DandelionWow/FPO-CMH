import argparse
import yaml
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch

from bank_init import BankInit
from online_learning import OnlineLearning
from eval import Eval
from utils import set_seed, load_clip_model, get_clip_category_feature, load_ucmh, MODEL_CATEGORY


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/config.yaml", type=str)
    args = parser.parse_args()
    assert os.path.isfile(args.cfg), "cfg file: {} not found".format(args.cfg)

    # merge config.yaml
    with open(args.cfg, 'r') as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)
    args_dict = vars(args)
    args_dict.update(yaml_config)
    args = argparse.Namespace(**args_dict)

    return args


def main():
    args = parse_args()

    # seed
    set_seed(args.seed)

    vocab_path = './data/categories_{}.txt'.format('imagenet')
    categories_feature_path = './data/categories_feature_{}.pkl'.format(args.dataset_name.lower())
    bank_path = './data/bank_{}_{}_{}.pkl'.format(args.dataset_name.lower(), args.model_name.lower(), str(args.code_len))
    category_idx_2_bank_idx_path = './data/category_idx_2_bank_idx_{}.pkl'.format(args.dataset_name.lower())

    # load ucmh
    imgNet, txtNet = load_ucmh(args.root, args.model_name, args.code_len, args.dataset_name)

    if args.mode == 'bank_init':
        # clip model
        tokenizer, _, clip_model = load_clip_model(args.clip_model)
        # Extract all category features
        categories, categories_feature = get_clip_category_feature(tokenizer, clip_model, args.template, vocab_path)
        
        bank_init = BankInit(args=args, imgNet=imgNet, txtNet=txtNet, categories=categories, categories_feature=categories_feature)
        bank_init.construct_bank()
        
        torch.save(categories_feature, categories_feature_path)
        torch.save(bank_init.bank, bank_path)
        torch.save(bank_init.category_idx_2_bank_idx, category_idx_2_bank_idx_path)
    elif args.mode == 'online_learning':
        bank = torch.load(bank_path)
        category_idx_2_bank_idx = torch.load(category_idx_2_bank_idx_path)
        categories_feature = torch.load(categories_feature_path)
        with open(vocab_path, 'r') as f:
            categories = [s.strip('\n') for s in f.readlines()]
        
        online_learning = OnlineLearning(args=args, bank=bank, imgNet=imgNet, txtNet=txtNet, categories=categories, categories_feature=categories_feature, category_idx_2_bank_idx=category_idx_2_bank_idx)
        online_learning.fpocmh()
        
        # save model param
        hash_model_param_dict = {
            'ImgNet': online_learning.imgNet.state_dict(),
            'TxtNet': online_learning.txtNet.state_dict(),
        }
        save_path = os.path.join(os.path.dirname(__file__), 'data', 'pretrained_model', MODEL_CATEGORY[args.model_name.lower()], args.model_name.lower(),)
        os.makedirs(save_path, exist_ok=True)
        torch.save(hash_model_param_dict, os.path.join(save_path, '{}_{}_{}bit_best_epoch.pth'.format(args.model_name.upper(), args.dataset_name, args.code_len)))
    elif args.mode == 'eval':
        # load ucmh models
        imgNet, txtNet = load_ucmh(os.path.join(os.path.dirname(__file__), 'data'), args.model_name, args.code_len, args.dataset_name)
        
        eval = Eval(args=args, imgNet=imgNet, txtNet=txtNet)
        if MODEL_CATEGORY[args.model_name.lower()] == 'ucmh':
            topk = 50
        else:
            topk = None
        eval.print_calculate_result(topk)


if __name__ == "__main__":
    main()
