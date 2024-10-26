import numpy as np

from torch.utils.data import Dataset

from feeders import tools
import torch
import torch.nn.functional as F


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path, allow_pickle=True)
        if self.split == 'train':
            self.data = npz_data['x_train']
            # y = npz_data['y_train']                                              # [[0. 0. 0. ... 0. 0. 1.], [1. 0. 0. ... 0. 0. 0.], [0. 1. 0. ... 0. 0. 0.], ..., [0. 0. 0. ... 0. 0. 0.]]
            # self.label = np.where(npz_data['y_train'] > 0)[1]  # 获取1对应的位置索引  [154   0   1   2   3   4   5   6   7   7   8   9  10  11  12  13  14  15 ... ]
            self.label = npz_data['y_train']

            index_to_remove = 13619

            # 使用 np.delete 删除指定索引的数据和标签
            self.data = np.delete(self.data, index_to_remove, axis=0)  # 在第一个维度上删除
            self.label = np.delete(self.label, index_to_remove, axis=0)  # 在第一个维度上删除

            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            # self.label = np.where(npz_data['y_test'] > 0)[1]
            self.label = npz_data['y_test']
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, _, _, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 17, 3)).transpose(0, 4, 1, 3, 2)

        # todo 比赛数据集格式转换 npy ---> npz{x_train, y_train, x_test, y_test}
        #   数据集格式：
        #      data:  (16523, 305, 102) --- (N, T, _)  ---> (N, T, 2, 17, 3) ---> (N, 3, T, 17, 2)
        #             最终需要的data纬度为  (N, 3, T, 17, 2)
        #     label:  (16523, 155)      --- (16523, )
        #             最终需要的label纬度为 (16523, )

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    # def __getitem__(self, index):
    #     data_numpy = self.data[index]
    #     label = self.label[index]
    #     data_numpy = np.array(data_numpy)
    #     valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
    #     # reshape Tx(MVC) to CTVM
    #     data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
    #     if self.random_rot:
    #         data_numpy = tools.random_rot(data_numpy)
    #     if self.bone:
    #         from .bone_pairs import ntu_pairs
    #         bone_data_numpy = np.zeros_like(data_numpy)
    #         for v1, v2 in ntu_pairs:
    #             bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
    #         data_numpy = bone_data_numpy
    #     if self.vel:
    #         data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
    #         data_numpy[:, -1] = 0
    #
    #     return data_numpy, label, index

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        # 计算有效帧数
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)

        # 调用 valid_crop_resize 并检查返回值是否为 None
        processed_data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)

        # 如果 valid_crop_resize 返回 None，使用原始的未处理数据
        if processed_data_numpy is None:
            print(f"Warning: Invalid cropping or resizing at index {index}. Using raw data.")
            processed_data_numpy = data_numpy  # 直接使用原始的 data_numpy

        # 如果数据帧数为 300，需要缩小到 64 帧
        if processed_data_numpy.shape[1] == 300:
            processed_data_numpy = self.downsample_to_64_frames(processed_data_numpy)

        # 如果启用了随机旋转
        if self.random_rot:
            processed_data_numpy = tools.random_rot(processed_data_numpy)

        # 如果启用了骨架数据计算
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(processed_data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = processed_data_numpy[:, :, v1 - 1] - processed_data_numpy[:, :, v2 - 1]
            processed_data_numpy = bone_data_numpy

        # 如果启用了速度计算
        if self.vel:
            processed_data_numpy[:, :-1] = processed_data_numpy[:, 1:] - processed_data_numpy[:, :-1]
            processed_data_numpy[:, -1] = 0

        return processed_data_numpy, label, index

    # 将300帧数据下采样至64帧的函数
    # def downsample_to_64_frames(self, data_numpy):
    #     # 获取数据的维度
    #     C, T, V, M = data_numpy.shape  # C: 通道数, T: 帧数, V: 关节数, M: 人数
    #
    #     # 将 numpy 数组转换为 torch 张量
    #     data_tensor = torch.tensor(data_numpy, dtype=torch.float32)
    #
    #     # 使用插值将帧数从300下采样到64
    #     data_tensor = data_tensor.permute(0, 3, 1, 2)  # 转换为 (C, M, T, V)
    #
    #     # 使用插值
    #     data_tensor = F.interpolate(data_tensor, size=self.window_size, mode='linear', align_corners=False)  # 下采样到64帧
    #
    #     data_tensor = data_tensor.permute(0, 2, 3, 1)  # 转换回 (C, T, V, M)
    #
    #     return data_tensor.numpy()

    def downsample_to_64_frames(self, data_numpy):
        # 获取数据的维度
        C, T, V, M = data_numpy.shape  # C: 通道数, T: 帧数, V: 关节数, M: 人数

        # 随机选择64帧的索引
        if T < 64:
            raise ValueError("帧数必须大于或等于64。")  # 确保帧数足够
        selected_indices = np.random.choice(T, 64, replace=False)  # 随机选择64个不同的帧索引

        # 按选定的索引选择帧
        downsampled_data = data_numpy[:, selected_indices, :, :]  # 选择随机的64帧

        return downsampled_data

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
