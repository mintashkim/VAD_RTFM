import torch.utils.data as data
import numpy as np
from utils import process_feat
import torch
from torch.utils.data import DataLoader
torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False, abnormal_cnt=None):
        self.modality = args.modality
        self.is_normal = is_normal
        self.dataset = args.dataset
        # if self.dataset == 'shanghai':
        #     if test_mode:
        #         self.rgb_list_file = 'list/shanghai-i3d-test-10crop.list'
        #     else:
        #         self.rgb_list_file = 'list/shanghai-i3d-train-10crop.list'
        # else:
        #     if test_mode:
        #         self.rgb_list_file = 'list/ucf-i3d-test.list'
        #     else:
        #         self.rgb_list_file = 'list/ucf-i3d.list'
        
        if test_mode:
            self.rgb_list_file = args.test_rgb_list
        else:
            self.rgb_list_file = args.rgb_list
        
        
        self.tranform = transform
        self.test_mode = test_mode
        self.abnormal_cnt = abnormal_cnt
        self._parse_list()
        self.num_frame = 0
        self.labels = None


    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.dataset == 'shanghai':
                if self.is_normal:
                    self.list = self.list[63:]
                    print('normal list for shanghai tech')
                    print(self.list)
                else:
                    self.list = self.list[:63]
                    print('abnormal list for shanghai tech')
                    print(self.list)

            else:
                abcnt = self.abnormal_cnt if self.abnormal_cnt else 810
                if self.is_normal:
                    self.list = self.list[abcnt:]
                    print(f'normal list for {self.dataset}')
                    print(self.list)
                    print("length:", len(self.list))
                else:
                    self.list = self.list[:abcnt]
                    print(f'abnormal list for {self.dataset}')
                    print(self.list)

    def __getitem__(self, index):

        label = self.get_label()  # get video level label 0/1
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features
        else:
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [10, B, T, F]
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)

            return divided_features, label

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
