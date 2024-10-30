import torch.utils.data as data
import torch

class BankInitDataset(data.Dataset):
    def __init__(self, image_features, text_features) -> None:
        super().__init__()
        self.image_features = torch.as_tensor(image_features, dtype=torch.float32)
        self.text_features = torch.as_tensor(text_features, dtype=torch.float32)


    def __getitem__(self, index):
        image_feature = self.image_features[index]
        text_feature = self.text_features[index]
        return image_feature, text_feature
    

    def __len__(self):
        return len(self.image_features)
    
class OnlineLearningDataset(data.Dataset):
    def __init__(self, image_features, text_features) -> None:
        super().__init__()
        streaming_data = torch.cat([torch.tensor(image_features), torch.tensor(text_features)], dim=0) # 2n * 512
        streaming_data_type = torch.zeros(size=(streaming_data.size()[0],), dtype=torch.int8)
        streaming_data_type[int(streaming_data.size()[0]/2):] = 1 # (0:5000)->(0,image) # (5000:10000)->(1,text)
        
        self.streaming_data_type = streaming_data_type
        self.streaming_data = streaming_data


    def __getitem__(self, index):
        return self.streaming_data[index, :], self.streaming_data_type[index]
    

    def __len__(self):
        return len(self.streaming_data)