import torch

from utils import zs_classifier


class StreamingData:
    ''' StreamingData entity class

        Attr:
            streaming_data: streaming_data feature (1, 512) \n
            streaming_data_type_str: 'text' or 'image'. \n
            is_image: Whether the modal is an image \n
            hash_model: Corresponding modal hash model \n
            streaming_data_rcode: streaming_data corresponds to the hash code \n
            clip_bank: Corresponding modal clip_bank \n
            code_bank: Corresponding modal code_bank \n
            inter_clip_bank: Inter-modal clip_bank \n
            inter_code_bank: Inter-modal code_bank \n
            streaming_data_categories: Categories of streaming_data \n
            streaming_data_weight: 0, 1, or 1-phi.
    '''
    def __init__(self, streaming_data, streaming_data_type, imgNet, txtNet, I_clip_bank, T_clip_bank, I_code_bank, T_code_bank, categories_feature, theta, category_idx_2_bank_idx, phi) -> None:
        self.streaming_data = streaming_data.cuda()
        # streaming_data modal
        self.is_image = True if streaming_data_type.item() == 0 else False
        self.hash_model = imgNet if self.is_image else txtNet
        self.streaming_data_type_str = 'image' if self.is_image else 'text'
        self.streaming_data_rcode = self.hash_model(self.streaming_data)
        with torch.no_grad():
            self.streaming_data_rcode_no_grad = self.hash_model(self.streaming_data)
        self.clip_bank = I_clip_bank.clone() if self.is_image else T_clip_bank.clone()
        self.inter_clip_bank = T_clip_bank.clone() if self.is_image else I_clip_bank.clone()
        self.code_bank = imgNet(self.clip_bank) if self.is_image else txtNet(self.clip_bank)
        self.inter_code_bank = txtNet(self.inter_clip_bank) if self.is_image else imgNet(self.inter_clip_bank)
        self.streaming_data_categories = zs_classifier(self.streaming_data, categories_feature, theta)
        if len(self.streaming_data_categories) == 0:
            self.streaming_data_weight = 0.
        else:
            self.streaming_data_weight = 1. - phi
            for streaming_data_category in self.streaming_data_categories:
                if streaming_data_category in category_idx_2_bank_idx.keys():
                    self.streaming_data_weight = 1.
                    break
        
        