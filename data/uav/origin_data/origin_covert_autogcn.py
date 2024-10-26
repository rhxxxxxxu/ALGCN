import numpy as np

# 加载npy文件
train_joint = np.load('train_joint.npy')
train_label = np.load('train_label.npy')
test_joint_A = np.load('test_joint_A.npy')
test_label_A = np.load('test_label_A.npy')

# 打印加载的数据形状，确保成功加载
print(f"train_joint shape: {train_joint.shape}")
print(f"train_label shape: {train_label.shape}")
print(f"test_joint_A shape: {test_joint_A.shape}")
print(f"test_label_A shape: {test_label_A.shape}")

# todo 原始数据纬度：
#       train_joint: (16432, 3, 300, 17, 2)
#       train_label: (16432, )                min:0  max:154
#       test_joint:  (2000, 3, 300, 17, 2)
#       train_label: (2000, )                 min:0  max:154


#todo 直接将原始数据data改为 Auto-GCN 中所需的格式 (label 除外), label 单独在代码中改

# 转换 train_joint 和 test_joint_A 的维度
# 从 (N, 3, 300, 17, 2) 转为 (N, 300, 2, 17, 3)  Auto-GCN 需要数据纬度为  (N, T, 2 , 17, 3)
train_joint_transposed = np.transpose(train_joint, (0, 2, 4, 3, 1))
test_joint_A_transposed = np.transpose(test_joint_A, (0, 2, 4, 3, 1))

# 打印转换后的数据形状
print(f"train_joint_transposed shape: {train_joint_transposed.shape}")
print(f"test_joint_A_transposed shape: {test_joint_A_transposed.shape}")

# 将转换后的数据保存为 npz 文件，包含训练和测试数据
np.savez('origin_data_converted.npz',
         x_train=train_joint_transposed,
         y_train=train_label,
         x_test=test_joint_A_transposed,
         y_test=test_label_A)

print("数据已成功保存为 'data_converted.npz'")