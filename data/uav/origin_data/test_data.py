import numpy as np

# 加载npy文件
test_joint = np.load('test_joint_B.npy')

# 打印加载的数据形状，确保成功加载
print(f"test_joint shape: {test_joint.shape}")

# 根据 test_joint_A 的长度生成随机标签，范围在 0 到 154 之间
num_samples = test_joint.shape[0]
test_label = np.random.randint(0, 155, size=num_samples)

# 转换 test_joint_A 的维度
# 从 (N, 3, 300, 17, 2) 转为 (N, 300, 2, 17, 3)  Auto-GCN 需要数据纬度为  (N, T, 2 , 17, 3)
test_joint_transposed = np.transpose(test_joint, (0, 2, 4, 3, 1))

# 打印转换后的数据形状
print(f"test_joint_transposed shape: {test_joint_transposed.shape}")

# 将转换后的数据保存为 npz 文件，包含测试数据和随机生成的标签
np.savez('test_data.npz',
         x_test=test_joint_transposed,
         y_test=test_label)

print("数据已成功保存为 'test_data.npz'")

