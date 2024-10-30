import torch
import torch.utils.data as data

from utils import calculate_top_map, compress, load_data


IMAGE = 'image'
TEXT = 'text'
LABEL = 'label'


class MyDataset(data.Dataset):
    def __init__(self, **kwargs):
        self.image_features = kwargs[IMAGE]
        self.text_features = kwargs[TEXT]
        self.labels = kwargs[LABEL]


    def __getitem__(self, index):
        image_feature = torch.as_tensor(self.image_features[index],dtype=torch.float32)
        text_feature = torch.as_tensor(self.text_features[index], dtype=torch.float32)
        label = torch.as_tensor(self.labels[index], dtype=torch.float32)
        return image_feature, text_feature, label


    def __len__(self):
        return len(self.labels)
    

class Eval():
    def __init__(self, args, imgNet, txtNet):
        self.args = args

        self.imgNet = imgNet.eval().cuda()
        self.txtNet = txtNet.eval().cuda()

        # load data
        feature, split = load_data(root=args.root, dataset_name=args.dataset_name)
        # online
        online_retrieval_index = torch.Tensor(split['online_retrieval_index'][:]).type(torch.int64).squeeze()
        online_test_index = torch.Tensor(split['online_test_index'][:]).type(torch.int64).squeeze()
        online_retrieval = {
            IMAGE:feature['online_image'][:].T[online_retrieval_index],
            TEXT:feature['online_text'][:].T[online_retrieval_index],
            LABEL:feature['online_label'][:].T[online_retrieval_index]
        }
        online_test = {
            IMAGE:feature['online_image'][:].T[online_test_index],
            TEXT:feature['online_text'][:].T[online_test_index],
            LABEL:feature['online_label'][:].T[online_test_index]
        }
        self.online_retrieval_dataloader = data.DataLoader(dataset=MyDataset(**online_retrieval), batch_size=args.eval_batch_size, shuffle=False)
        self.online_test_dataloader = data.DataLoader(dataset=MyDataset(**online_test), batch_size=args.eval_batch_size, shuffle=False)
        # offline
        offline_retrieval_index = torch.Tensor(split['offline_retrieval_index'][:]).type(torch.int64).squeeze()
        offline_test_index = torch.Tensor(split['offline_test_index'][:]).type(torch.int64).squeeze()
        offline_retrieval = {
            IMAGE:feature['offline_image'][:].T[offline_retrieval_index],
            TEXT:feature['offline_text'][:].T[offline_retrieval_index],
            LABEL:feature['offline_label'][:].T[offline_retrieval_index]
        }
        offline_test = {
            IMAGE:feature['offline_image'][:].T[offline_test_index],
            TEXT:feature['offline_text'][:].T[offline_test_index],
            LABEL:feature['offline_label'][:].T[offline_test_index]
        }
        self.offline_retrieval_dataloader = data.DataLoader(dataset=MyDataset(**offline_retrieval), batch_size=args.eval_batch_size, shuffle=False)
        self.offline_test_dataloader = data.DataLoader(dataset=MyDataset(**offline_test), batch_size=args.eval_batch_size, shuffle=False)


    def print_calculate_result(self, topk):
        list = ['offline', 'online',]

        for str in list:
            retrieval_loader = getattr(self, str + '_retrieval_dataloader')
            query_loader = getattr(self, str + '_test_dataloader')
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(retrieval_loader, query_loader, self.imgNet, self.txtNet)
            MAP_I2T = calculate_top_map(qu_BI, re_BT, qu_L, re_L, topk=topk)
            MAP_T2I = calculate_top_map(qu_BT, re_BI, qu_L, re_L, topk=topk)
            MAP_I2I = calculate_top_map(qu_BI, re_BI, qu_L, re_L, topk=topk)
            MAP_T2T = calculate_top_map(qu_BT, re_BT, qu_L, re_L, topk=topk)

            print('[{} mAP]: '.format(str) + '(I->T): %3.4f, (T->I): %3.4f, (I->I): %3.4f, (T->T): %3.4f' % (MAP_I2T, MAP_T2I, MAP_I2I, MAP_T2T))

        print('\n')
