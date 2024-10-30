import torch
import torch.nn.functional as F

from tqdm import tqdm

from utils import load_data, load_dataloader
from streaming_data import StreamingData


class OnlineLearning():
    def __init__(self, args, imgNet, txtNet, bank: dict, categories, categories_feature, category_idx_2_bank_idx):
        self.args = args

        self.I_clip_bank = bank['I_clip_bank'].cuda()
        self.T_clip_bank = bank['T_clip_bank'].cuda()
        self.I_code_bank = bank['I_code_bank'].cuda()
        self.T_code_bank = bank['T_code_bank'].cuda()

        self.categories = categories
        self.categories_feature = categories_feature.cuda()
        self.category_idx_2_bank_idx = category_idx_2_bank_idx

        self.imgNet = imgNet.train().cuda()
        self.txtNet = txtNet.train().cuda()

        self.offline_bank_len = len(self.I_clip_bank)
        self.I_clip_bank_offline = self.I_clip_bank.clone()
        self.T_clip_bank_offline = self.T_clip_bank.clone()
        with torch.no_grad():
            self.I_code_bank_offline = self.imgNet(self.I_clip_bank_offline)
            self.T_code_bank_offline = self.txtNet(self.T_clip_bank_offline)

        # load dataset
        assert self.args.shuffle is True, "During the online_learning, shuffle must be True"
        assert self.args.batch_size == 1, "During the online_learning, batch_size must be 1"
        assert self.args.phase == 'online', "During the online_learning, phase must be online"
        feature, split = load_data(root=args.root, dataset_name=args.dataset_name)
        self.dataloader = load_dataloader(feature=feature, split=split, mode=args.mode, phase=self.args.phase, batch_size=args.batch_size, shuffle=args.shuffle)

        self.opt_hash_model_I, self.opt_hash_model_T = self._load_optim()
    

    def fpocmh(self):
        self.novel_categories = []
        self.clip_dict = {
            'image':{},
            'text':{},
        }
        self.code_dict = {
            'image':{},
            'text':{},
        }
        self.clip_update_counter_dict = {
            'image':{},
            'text':{},
        }
        streaming_data_count = {
            'image': 0,
            'text': 0,
        }

        for epoch in range(self.args.epochs):
            for i, (streaming_data, t) in tqdm(enumerate(self.dataloader)):
                streaming_data = self._get_streaming_data_dict(streaming_data, t)
                if streaming_data.is_image:
                    streaming_data_count['image'] += 1
                else:
                    streaming_data_count['text'] += 1
                
                # streaming partial-modal learning loss
                self._calc_streaming_loss(streaming_data=streaming_data)
                
                # update bank
                self._update_bank(streaming_data=streaming_data)

                # initial-anchor rehearsal loss
                self._calc_rehearsal_loss(streaming_data_count=streaming_data_count)

                # step
                if streaming_data_count['image'] % self.args.mu == 0 and streaming_data_count['image'] != 0:
                    streaming_data_count['image'] = 0
                    self.opt_hash_model_I.step()
                    self.opt_hash_model_I.zero_grad()
                elif streaming_data_count['text'] % self.args.mu == 0 and streaming_data_count['text'] != 0:
                    streaming_data_count['text'] = 0
                    self.opt_hash_model_T.step()
                    self.opt_hash_model_T.zero_grad()
                else:
                    pass
                

    def _update_bank(self, streaming_data: StreamingData):
        # Iterate over the category of streaming_data
        for category_idx in streaming_data.streaming_data_categories:
            clazz = self.categories[category_idx]
            bank_idx = self.category_idx_2_bank_idx.get(category_idx)
            
            if bank_idx is None:
                # get centre
                clip_centre = self.clip_dict[streaming_data.streaming_data_type_str].get(category_idx)
                code_centre = self.code_dict[streaming_data.streaming_data_type_str].get(category_idx)
                # is None -> init centre
                if (clip_centre is None) or (code_centre is None):
                    clip_centre = self._init_clip_centre(streaming_data.streaming_data)
                    code_centre = self._init_code_centre(streaming_data.streaming_data, streaming_data.hash_model)
                    self.clip_update_counter_dict[streaming_data.streaming_data_type_str][category_idx] = 0
                # calc centre
                clip_centre = torch.mean(torch.cat([streaming_data.streaming_data, clip_centre]), dim=0, keepdim=True)
                code_centre = torch.mean(torch.cat([streaming_data.streaming_data_rcode_no_grad, code_centre]), dim=0, keepdim=True)
                # update 
                self.clip_dict[streaming_data.streaming_data_type_str][category_idx] = clip_centre
                self.code_dict[streaming_data.streaming_data_type_str][category_idx] = code_centre
                # count
                self.clip_update_counter_dict[streaming_data.streaming_data_type_str][category_idx] += 1

                # is enable
                count_image = self.clip_update_counter_dict['image'].get(category_idx)
                count_text = self.clip_update_counter_dict['text'].get(category_idx)
                if (count_image and count_text) \
                    and (count_image >= self.args.is_accessible_bank_count) \
                    and (count_text >= self.args.is_accessible_bank_count) \
                    and (bank_idx is None):
                    # Update to bank
                    self.I_clip_bank = torch.cat([self.I_clip_bank, self.clip_dict['image'].get(category_idx)])
                    self.I_code_bank = torch.cat([self.I_code_bank, self.code_dict['image'].get(category_idx)])
                    self.T_clip_bank = torch.cat([self.T_clip_bank, self.clip_dict['text'].get(category_idx)])
                    self.T_code_bank = torch.cat([self.T_code_bank, self.code_dict['text'].get(category_idx)])
                    # save idx
                    self.category_idx_2_bank_idx[category_idx] = len(self.I_clip_bank) - 1 # 0-based
                    self.novel_categories.append(clazz)
            else:
                clip_centre = streaming_data.clip_bank[bank_idx].unsqueeze(0)
                code_centre = streaming_data.code_bank[bank_idx].unsqueeze(0)
                clip_centre = torch.mean(torch.cat([streaming_data.streaming_data, clip_centre]), dim=0, keepdim=True)
                code_centre = torch.mean(torch.cat([streaming_data.streaming_data_rcode_no_grad, code_centre]), dim=0, keepdim=True)
            
            # The enabled centre is updated to the bank
            if bank_idx:
                if streaming_data.is_image:
                    self.I_clip_bank[bank_idx] = clip_centre.squeeze(0)
                    self.I_code_bank[bank_idx] = code_centre.squeeze(0)
                else:
                    self.T_clip_bank[bank_idx] = clip_centre.squeeze(0)
                    self.T_code_bank[bank_idx] = code_centre.squeeze(0)


    def _get_streaming_data_dict(self, streaming_data, streaming_data_type):
        return StreamingData(
            streaming_data = streaming_data,
            streaming_data_type = streaming_data_type,
            imgNet = self.imgNet,
            txtNet = self.txtNet,
            I_clip_bank = self.I_clip_bank,
            T_clip_bank = self.T_clip_bank,
            I_code_bank = self.I_code_bank,
            T_code_bank = self.T_code_bank,
            categories_feature = self.categories_feature,
            theta = self.args.theta,
            category_idx_2_bank_idx = self.category_idx_2_bank_idx,
            phi = self.args.phi
        )


    def _calc_streaming_loss(self, streaming_data: StreamingData):
        intra_loss = F.cross_entropy(self._calc_sim(streaming_data.streaming_data_rcode, streaming_data.code_bank), \
                                     self._calc_logits(streaming_data.streaming_data, streaming_data.clip_bank))
        inter_loss = F.cross_entropy(self._calc_sim(streaming_data.streaming_data_rcode, streaming_data.inter_code_bank), \
                                     self._calc_logits(streaming_data.streaming_data, streaming_data.inter_clip_bank))
        
        streaming_loss = self.args.alpha * intra_loss + self.args.beta * inter_loss
        streaming_loss = streaming_data.streaming_data_weight * streaming_loss / self.args.mu
        streaming_loss.backward()


    def _calc_rehearsal_loss(self, streaming_data_count: dict):
        if streaming_data_count['image'] % self.args.mu == 0 and streaming_data_count['image'] != 0:
            I_code_bank = self.imgNet(self.I_clip_bank_offline)
            sign_loss = F.mse_loss(I_code_bank, torch.sign(self.I_code_bank_offline))
        elif streaming_data_count['text'] % self.args.mu == 0 and streaming_data_count['text'] != 0:
            T_code_bank = self.txtNet(self.T_clip_bank_offline)
            sign_loss = F.mse_loss(T_code_bank, torch.sign(self.T_code_bank_offline))
        else:
            return

        rehearsal_loss = sign_loss
        rehearsal_loss.backward()     


    def _init_clip_centre(self, x:torch.Tensor):
        if len(x.size()) == 1:
            x = x.unsqueeze(0)
        return x
    

    @torch.no_grad()
    def _init_code_centre(self, x:torch.Tensor, model):
        if len(x.size()) == 1:
            x = x.unsqueeze(0)
        return model(x.cuda())


    @torch.no_grad()
    def _forward_with_no_grad(self, x:torch.Tensor, model):
        return model(x.cuda())


    def _calc_sim(self, x, y):
        return F.normalize(x) @ F.normalize(y).T
    
    
    def _calc_logits(self, x, y):
        sim = self._calc_sim(x, y)
        logits = (100. * sim).softmax(dim=-1)
        return logits


    def _calc_dist(self, x, y):
        return torch.cdist(x, y)


    def is_streaming_data_known_category(self, list):
        if len(list) == 0:
            return False
        for item in list:
            bank_idx = self.category_idx_2_bank_idx.get(item)
            if bank_idx is None:
                return False
        return True


    def _list_2_str(self, target: list):
        target = [str(t) for t in target]
        return ','.join(target)
    

    def _load_optim(self):
        if self.args.optim == 'adam':
            opt_hash_model_I = torch.optim.Adam(self.imgNet.parameters(), lr=self.args.lr_image)
            opt_hash_model_T = torch.optim.Adam(self.txtNet.parameters(), lr=self.args.lr_text)
        elif self.args.optim == 'adamw':
            opt_hash_model_I = torch.optim.AdamW(self.imgNet.parameters(), lr=self.args.lr_image)
            opt_hash_model_T = torch.optim.AdamW(self.txtNet.parameters(), lr=self.args.lr_text)
        elif self.args.optim == 'sgd':
            opt_hash_model_I = torch.optim.SGD(self.imgNet.parameters(), lr=self.args.lr_image, momentum=0.9, weight_decay=0.0005)
            opt_hash_model_T = torch.optim.SGD(self.txtNet.parameters(), lr=self.args.lr_text, momentum=0.9, weight_decay=0.0005)
        else:
            opt_hash_model_I = torch.optim.SGD(self.imgNet.parameters(), lr=self.args.lr_image)
            opt_hash_model_T = torch.optim.SGD(self.txtNet.parameters(), lr=self.args.lr_text)
        return opt_hash_model_I, opt_hash_model_T