import numpy as np
import torch
import torch.nn.functional as F


class ChannelDataAugmentation:
    """信道数据增强类"""

    def __init__(self, noise_std=0.01, rotation_range=0.1, enable_mixup=True):
        self.noise_std = noise_std
        self.rotation_range = rotation_range
        self.enable_mixup = enable_mixup

    def add_gaussian_noise(self, data, std=None):
        """添加高斯噪声"""
        if std is None:
            std = self.noise_std
        noise = np.random.normal(0, std, data.shape)
        return data + noise

    def rotate_channel_matrix(self, data):
        """随机旋转信道矩阵（通过相位旋转实现）"""
        # data: [M, N, 2] 实虚部格式
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)

        # 转换为复数
        complex_data = data[..., 0] + 1j * data[..., 1]

        # 应用相位旋转
        rotation_factor = np.exp(1j * angle)
        rotated_data = complex_data * rotation_factor

        # 转换回实虚部格式
        result = np.stack([rotated_data.real, rotated_data.imag], axis=-1)
        return result.astype(data.dtype)

    def mixup(self, data1, label1, data2, label2, alpha=0.2):
        """Mixup数据增强"""
        lam = np.random.beta(alpha, alpha)
        mixed_data = lam * data1 + (1 - lam) * data2
        mixed_label = lam * label1 + (1 - lam) * label2
        return mixed_data, mixed_label, lam

    def augment_sample(self, data, label=None, augment_prob=0.5):
        """对单个样本进行增强"""
        augmented_data = data.copy()
        augmented_label = label.copy() if label is not None else None

        # 随机决定是否应用增强
        if np.random.random() < augment_prob:
            # 添加噪声
            if np.random.random() < 0.3:
                noise_std = np.random.uniform(0.005, self.noise_std)
                augmented_data = self.add_gaussian_noise(augmented_data, noise_std)
                if augmented_label is not None:
                    augmented_label = self.add_gaussian_noise(augmented_label, noise_std * 0.5)

            # 相位旋转
            if np.random.random() < 0.2:
                augmented_data = self.rotate_channel_matrix(augmented_data)
                if augmented_label is not None:
                    augmented_label = self.rotate_channel_matrix(augmented_label)

        return augmented_data, augmented_label


class ChannelDataPreprocessor:
    """信道数据预处理类"""

    def __init__(self):
        self.data_stats = None
        self.label_stats = None

    def compute_statistics(self, data, labels=None):
        """计算数据统计信息"""
        # 计算均值和标准差
        self.data_stats = {
            'mean': np.mean(data, axis=(0, 1, 2), keepdims=True),
            'std': np.std(data, axis=(0, 1, 2), keepdims=True) + 1e-8,
            'min': np.min(data),
            'max': np.max(data)
        }

        if labels is not None:
            self.label_stats = {
                'mean': np.mean(labels, axis=(0, 1, 2), keepdims=True),
                'std': np.std(labels, axis=(0, 1, 2), keepdims=True) + 1e-8,
                'min': np.min(labels),
                'max': np.max(labels)
            }

        print("Data statistics computed:")
        print(f"  Data: mean={self.data_stats['mean'].mean():.4f}, std={self.data_stats['std'].mean():.4f}")
        if self.label_stats:
            print(f"  Labels: mean={self.label_stats['mean'].mean():.4f}, std={self.label_stats['std'].mean():.4f}")

    def normalize_data(self, data, use_robust=True):
        """归一化数据"""
        if self.data_stats is None:
            raise ValueError("Statistics not computed. Call compute_statistics first.")

        if use_robust:
            # 使用鲁棒归一化（基于分位数）
            q25, q75 = np.percentile(data, [25, 75])
            iqr = q75 - q25 + 1e-8
            median = np.median(data)
            normalized = (data - median) / iqr
        else:
            # 标准Z-score归一化
            normalized = (data - self.data_stats['mean']) / self.data_stats['std']

        return normalized

    def normalize_labels(self, labels):
        """归一化标签"""
        if self.label_stats is None:
            raise ValueError("Label statistics not computed.")

        normalized = (labels - self.label_stats['mean']) / self.label_stats['std']
        return normalized

    def denormalize_labels(self, normalized_labels):
        """反归一化标签"""
        if self.label_stats is None:
            raise ValueError("Label statistics not computed.")

        denormalized = normalized_labels * self.label_stats['std'] + self.label_stats['mean']
        return denormalized

    def remove_outliers(self, data, labels=None, threshold=3.0):
        """移除异常值"""
        # 计算每个样本的范数
        norms = np.linalg.norm(data.reshape(data.shape[0], -1), axis=1)

        # 使用3-sigma规则检测异常值
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)

        mask = np.abs(norms - mean_norm) < threshold * std_norm

        print(f"Removing {np.sum(~mask)} outliers out of {len(data)} samples")

        if labels is not None:
            return data[mask], labels[mask], mask
        else:
            return data[mask], mask


def create_data_splits(data, labels, val_ratio=0.1, test_ratio=0.0, random_seed=42):
    """创建数据分割"""
    np.random.seed(random_seed)

    n_samples = len(data)
    indices = np.random.permutation(n_samples)

    # 计算分割点
    val_size = int(n_samples * val_ratio)
    test_size = int(n_samples * test_ratio)
    train_size = n_samples - val_size - test_size

    # 分割索引
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]

    splits = {
        'train_data': data[train_indices],
        'train_labels': labels[train_indices],
        'val_data': data[val_indices],
        'val_labels': labels[val_indices],
        'train_indices': train_indices,
        'val_indices': val_indices
    }

    if test_size > 0:
        test_indices = indices[train_size + val_size:]
        splits.update({
            'test_data': data[test_indices],
            'test_labels': labels[test_indices],
            'test_indices': test_indices
        })

    print(f"Data splits created:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val: {len(val_indices)} samples")
    if test_size > 0:
        print(f"  Test: {len(test_indices)} samples")

    return splits


def analyze_channel_properties(data, labels=None):
    """分析信道特性"""
    print("Analyzing channel properties...")

    # 基本统计
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Data range: [{np.min(data):.4f}, {np.max(data):.4f}]")

    # 转换为复数进行分析
    data_complex = data[..., 0] + 1j * data[..., 1]

    # 条件数分析
    condition_numbers = []
    rank_deficient_count = 0

    for i in range(min(1000, len(data))):  # 分析前1000个样本
        try:
            U, S, Vh = np.linalg.svd(data_complex[i])
            cond_num = S[0] / (S[-1] + 1e-12)
            condition_numbers.append(cond_num)

            if S[-1] < 1e-10:
                rank_deficient_count += 1
        except:
            continue

    if condition_numbers:
        print(f"Condition number statistics:")
        print(f"  Mean: {np.mean(condition_numbers):.2f}")
        print(f"  Std: {np.std(condition_numbers):.2f}")
        print(f"  Range: [{np.min(condition_numbers):.2f}, {np.max(condition_numbers):.2f}]")
        print(f"  Rank deficient matrices: {rank_deficient_count}")

    # 奇异值分布
    if len(data) > 0:
        sample_svd = np.linalg.svd(data_complex[0], compute_uv=False)
        print(f"Sample singular values (first matrix): {sample_svd[:10]}")

    if labels is not None:
        labels_complex = labels[..., 0] + 1j * labels[..., 1]
        print(f"Labels shape: {labels.shape}")
        print(f"Labels range: [{np.min(labels):.4f}, {np.max(labels):.4f}]")

        # 计算重构误差统计
        recon_errors = []
        for i in range(min(100, len(data))):
            error = np.linalg.norm(data_complex[i] - labels_complex[i], 'fro') / \
                    np.linalg.norm(labels_complex[i], 'fro')
            recon_errors.append(error)

        if recon_errors:
            print(f"Reconstruction error (data vs labels):")
            print(f"  Mean: {np.mean(recon_errors):.4f}")
            print(f"  Std: {np.std(recon_errors):.4f}")


class AdaptiveDataLoader:
    """自适应数据加载器"""

    def __init__(self, data, labels, batch_size=4, shuffle=True, augment=False):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

        if augment:
            self.augmenter = ChannelDataAugmentation()

        self.indices = np.arange(len(data))
        if shuffle:
            np.random.shuffle(self.indices)

        self.current_pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_pos >= len(self.indices):
            # 重置并重新打乱
            self.current_pos = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
            raise StopIteration

        # 获取批次索引
        end_pos = min(self.current_pos + self.batch_size, len(self.indices))
        batch_indices = self.indices[self.current_pos:end_pos]
        self.current_pos = end_pos

        # 准备批次数据
        batch_data = []
        batch_labels = []

        for idx in batch_indices:
            data_sample = self.data[idx].copy()
            label_sample = self.labels[idx].copy()

            # 应用数据增强
            if self.augment:
                data_sample, label_sample = self.augmenter.augment_sample(
                    data_sample, label_sample
                )

            batch_data.append(data_sample)
            batch_labels.append(label_sample)

        return np.array(batch_data), np.array(batch_labels), batch_indices

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size