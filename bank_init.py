import torch

from utils import zs_classifier, load_data, load_dataloader

class BankInit():
    def __init__(self, args, imgNet, txtNet, categories, categories_feature):
        self.args = args
        
        self.categories = categories
        self.categories_feature = categories_feature
        self.category_idx_2_bank_idx = {}

        # load data
        assert self.args.shuffle is False, "During the bank_init, shuffle must be False"
        assert self.args.phase == 'offline', "During the bank_init, phase must be offline"
        feature, split = load_data(root=args.root, dataset_name=args.dataset_name)
        self.dataloader = load_dataloader(feature=feature, split=split, mode=args.mode, phase=self.args.phase, batch_size=args.batch_size, shuffle=args.shuffle)

        # relaxed hash code
        with torch.no_grad():
            self.code_I = imgNet(self.dataloader.dataset.image_features.cuda())
            self.code_T = txtNet(self.dataloader.dataset.text_features.cuda())
    

    def construct_bank(self):
        image_features = self.dataloader.dataset.image_features
        text_features = self.dataloader.dataset.text_features
        code_I = self.code_I
        code_T = self.code_T
        
        # anno
        offline_categories = []
        # image
        I_offline_categories = set()
        I_plabels = torch.zeros(len(image_features), len(self.categories))
        for idx, image_feature in enumerate(image_features):
            top = zs_classifier(image_feature, self.categories_feature, self.args.theta)
            if len(top) == 0:
                continue
            I_plabels[idx, top] = 1
            for i in top:
                I_offline_categories.add(i)
        # text
        T_offline_categories = set()
        T_plabels = torch.zeros(len(text_features), len(self.categories))
        for idx, text_feature in enumerate(text_features):
            top = zs_classifier(text_feature, self.categories_feature, self.args.theta)
            if len(top) == 0:
                continue
            T_plabels[idx, top] = 1
            for i in top:
                T_offline_categories.add(i)
        offline_categories = list(I_offline_categories & T_offline_categories)
        
        # init bank
        # image
        I_clip_bank = []
        I_code_bank = []
        for idx in offline_categories:
            samples_idx = torch.where(I_plabels[:, idx] == 1)[0]
            I_clip_bank.append(torch.mean(image_features[samples_idx], dim=0))
            I_code_bank.append(torch.mean(code_I[samples_idx], dim=0))
            self.category_idx_2_bank_idx[idx] = len(I_clip_bank) - 1
        I_clip_bank = torch.stack(I_clip_bank) # [offline_categories_num, 512]
        I_code_bank = torch.stack(I_code_bank) # [offline_categories_num, 16]
        # text
        T_clip_bank = []
        T_code_bank = []
        for idx in offline_categories:
            samples_idx = torch.where(T_plabels[:, idx] == 1)[0]
            T_clip_bank.append(torch.mean(text_features[samples_idx], dim=0))
            T_code_bank.append(torch.mean(code_T[samples_idx], dim=0))
        T_clip_bank = torch.stack(T_clip_bank) # [offline_categories_num, 512]
        T_code_bank = torch.stack(T_code_bank) # [offline_categories_num, 16]

        self.bank = {
            'I_clip_bank': I_clip_bank,
            'T_clip_bank': T_clip_bank,
            'I_code_bank': I_code_bank,
            'T_code_bank': T_code_bank,
        }