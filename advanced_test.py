import os
import numpy as np
import torch
import torch.nn.functional as F
from solution import SVDNet, compute_ae_metric
from data_utils import ChannelDataPreprocessor, analyze_channel_properties
from tqdm import tqdm
import matplotlib.pyplot as plt


def read_cfg_file(file_path):
    """读取配置文件"""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    samp_num = int(lines[0].strip())
    M = int(lines[1].strip())
    N = int(lines[2].strip())
    IQ = int(lines[3].strip())
    R = int(lines[4].strip())

    return samp_num, M, N, IQ, R


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()

    def evaluate_single_sample(self, data, label=None):
        """评估单个样本"""
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data).to(self.device)
            U_out, S_out, V_out = self.model(data_tensor)

            results = {
                'U': U_out.cpu().numpy(),
                'S': S_out.cpu().numpy(),
                'V': V_out.cpu().numpy()
            }

            if label is not None:
                label_tensor = torch.FloatTensor(label).to(self.device)
                ae = compute_ae_metric(U_out, S_out, V_out, label_tensor)
                results['ae'] = ae

                # 计算详细的误差分析
                results.update(self._detailed_error_analysis(U_out, S_out, V_out, label_tensor))

            return results

    def _detailed_error_analysis(self, U, S, V, H_ideal):
        """详细误差分析"""
        # 转换为复数
        U_complex = U[..., 0] + 1j * U[..., 1]
        V_complex = V[..., 0] + 1j * V[..., 1]
        H_ideal_complex = H_ideal[..., 0] + 1j * H_ideal[..., 1]

        # 重构矩阵
        S_expanded = S.unsqueeze(0).expand(U_complex.shape[0], -1)
        U_scaled = U_complex * S_expanded
        H_recon = U_scaled @ V_complex.conj().T

        # 各种误差指标
        frobenius_error = torch.norm(H_ideal_complex - H_recon, p='fro') / torch.norm(H_ideal_complex, p='fro')

        spectral_error = torch.norm(H_ideal_complex - H_recon, p=2) / torch.norm(H_ideal_complex, p=2)

        # 正交性误差
        I = torch.eye(S.shape[0], device=U.device, dtype=torch.complex64)
        U_ortho_error = torch.norm(U_complex.conj().T @ U_complex - I, p='fro')
        V_ortho_error = torch.norm(V_complex.conj().T @ V_complex - I, p='fro')

        # 奇异值误差
        true_svd = torch.linalg.svd(H_ideal_complex, compute_uv=False)
        true_S = true_svd[:S.shape[0]]  # 取前R个
        singular_error = torch.norm(S - true_S) / torch.norm(true_S)

        return {
            'frobenius_error': frobenius_error.item(),
            'spectral_error': spectral_error.item(),
            'U_ortho_error': U_ortho_error.item(),
            'V_ortho_error': V_ortho_error.item(),
            'singular_error': singular_error.item()
        }

    def evaluate_batch(self, data_batch, label_batch=None):
        """评估一个批次"""
        batch_results = []

        for i in range(len(data_batch)):
            data = data_batch[i]
            label = label_batch[i] if label_batch is not None else None

            result = self.evaluate_single_sample(data, label)
            batch_results.append(result)

        return batch_results

    def compute_model_complexity(self):
        """计算模型复杂度"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # 估算FLOPs（浮点运算次数）
        # 这是一个简化的估算
        flops = 0
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                flops += module.in_features * module.out_features
            elif isinstance(module, torch.nn.Conv2d):
                flops += module.in_channels * module.out_channels * \
                         module.kernel_size[0] * module.kernel_size[1]

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'estimated_flops': flops
        }


def test_model_comprehensive(round_idx=1, scene_idx=1, use_validation=False):
    """全面测试模型"""
    print(f"Comprehensive Testing - Scene {scene_idx} of Round {round_idx}")

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据路径
    data_dir = f"./CompetitionData{round_idx}"
    cfg_path = f"{data_dir}/Round{round_idx}CfgData{scene_idx}.txt"
    test_data_path = f"{data_dir}/Round{round_idx}TestData{scene_idx}.npy"

    # 读取配置
    samp_num, M, N, IQ, R = read_cfg_file(cfg_path)
    print(f"Config: samp_num={samp_num}, M={M}, N={N}, R={R}")

    # 创建并加载模型
    model = SVDNet(M=M, N=N, R=R).to(device)

    model_path = f"svd_model_round{round_idx}_scene{scene_idx}.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found, using random weights")

    # 加载数据
    test_data = np.load(test_data_path)
    actual_samp_num = test_data.shape[0]
    print(f"Test data shape: {test_data.shape}")

    # 如果是验证模式，也加载训练数据进行对比
    validation_data = None
    if use_validation:
        train_data_path = f"{data_dir}/Round{round_idx}TrainData{scene_idx}.npy"
        train_label_path = f"{data_dir}/Round{round_idx}TrainLabel{scene_idx}.npy"

        if os.path.exists(train_data_path) and os.path.exists(train_label_path):
            validation_data = {
                'data': np.load(train_data_path)[:100],  # 取前100个样本验证
                'labels': np.load(train_label_path)[:100]
            }
            print("Validation data loaded for comparison")

    # 分析数据特性
    analyze_channel_properties(test_data)

    # 创建评估器
    evaluator = ModelEvaluator(model, device)

    # 计算模型复杂度
    complexity = evaluator.compute_model_complexity()
    print(f"Model complexity:")
    print(f"  Parameters: {complexity['total_params']:,}")
    print(f"  Estimated FLOPs: {complexity['estimated_flops']:,}")

    # 准备输出数组
    U_out_all = np.zeros((actual_samp_num, M, R, IQ), dtype=np.float32)
    S_out_all = np.zeros((actual_samp_num, R), dtype=np.float32)
    V_out_all = np.zeros((actual_samp_num, N, R, IQ), dtype=np.float32)

    # 如果有验证数据，收集详细统计
    detailed_stats = []

    print("Processing test samples...")
    batch_size = 8

    for batch_start in tqdm(range(0, actual_samp_num, batch_size)):
        batch_end = min(batch_start + batch_size, actual_samp_num)

        for samp_idx in range(batch_start, batch_end):
            # 处理单个样本
            data_sample = test_data[samp_idx]

            # 评估
            result = evaluator.evaluate_single_sample(data_sample)

            # 保存结果
            U_out_all[samp_idx] = result['U']
            S_out_all[samp_idx] = result['S']
            V_out_all[samp_idx] = result['V']

    # 验证模式的详细分析
    if validation_data is not None:
        print("\nValidation analysis...")
        val_ae_scores = []
        val_detailed_stats = []

        for i in range(len(validation_data['data'])):
            result = evaluator.evaluate_single_sample(
                validation_data['data'][i],
                validation_data['labels'][i]
            )
            val_ae_scores.append(result['ae'])
            val_detailed_stats.append({
                'frobenius_error': result['frobenius_error'],
                'spectral_error': result['spectral_error'],
                'U_ortho_error': result['U_ortho_error'],
                'V_ortho_error': result['V_ortho_error'],
                'singular_error': result['singular_error']
            })

        # 打印验证统计
        print(f"Validation Results (n={len(val_ae_scores)}):")
        print(f"  AE Score: {np.mean(val_ae_scores):.4f} ± {np.std(val_ae_scores):.4f}")
        print(f"  Frobenius Error: {np.mean([s['frobenius_error'] for s in val_detailed_stats]):.4f}")
        print(f"  Spectral Error: {np.mean([s['spectral_error'] for s in val_detailed_stats]):.4f}")
        print(f"  U Orthogonality: {np.mean([s['U_ortho_error'] for s in val_detailed_stats]):.4f}")
        print(f"  V Orthogonality: {np.mean([s['V_ortho_error'] for s in val_detailed_stats]):.4f}")
        print(f"  Singular Value Error: {np.mean([s['singular_error'] for s in val_detailed_stats]):.4f}")

    # 输出质量检查
    print("\nOutput quality analysis...")

    # 奇异值统计
    print(f"Singular values statistics:")
    print(f"  Mean: {np.mean(S_out_all):.4f}")
    print(f"  Std: {np.std(S_out_all):.4f}")
    print(f"  Range: [{np.min(S_out_all):.4f}, {np.max(S_out_all):.4f}]")

    # 检查奇异值降序性
    descent_violations = 0
    for i in range(actual_samp_num):
        if not np.all(S_out_all[i][:-1] >= S_out_all[i][1:]):
            descent_violations += 1

    print(f"Singular value descent violations: {descent_violations}/{actual_samp_num}")

    # 正交性检查（抽样）
    ortho_errors_U = []
    ortho_errors_V = []

    sample_indices = np.random.choice(actual_samp_num, min(100, actual_samp_num), replace=False)

    for idx in sample_indices:
        U_sample = U_out_all[idx]
        V_sample = V_out_all[idx]

        # 转换为复数
        U_complex = U_sample[..., 0] + 1j * U_sample[..., 1]
        V_complex = V_sample[..., 0] + 1j * V_sample[..., 1]

        # 计算正交性误差
        I_U = np.eye(R)
        I_V = np.eye(R)

        U_gram = np.conj(U_complex.T) @ U_complex
        V_gram = np.conj(V_complex.T) @ V_complex

        ortho_error_U = np.linalg.norm(U_gram - I_U, 'fro')
        ortho_error_V = np.linalg.norm(V_gram - I_V, 'fro')

        ortho_errors_U.append(ortho_error_U)
        ortho_errors_V.append(ortho_error_V)

    print(f"Orthogonality errors (sampled):")
    print(f"  U matrices: {np.mean(ortho_errors_U):.4f} ± {np.std(ortho_errors_U):.4f}")
    print(f"  V matrices: {np.mean(ortho_errors_V):.4f} ± {np.std(ortho_errors_V):.4f}")

    # 保存结果
    output_file = f"Round{round_idx}TestOutput{scene_idx}.npz"
    np.savez(output_file, U_out=U_out_all, S_out=S_out_all, V_out=V_out_all)
    print(f"\nResults saved to {output_file}")

    # 保存详细分析结果
    analysis_results = {
        'complexity': complexity,
        'singular_stats': {
            'mean': np.mean(S_out_all),
            'std': np.std(S_out_all),
            'min': np.min(S_out_all),
            'max': np.max(S_out_all)
        },
        'orthogonality_stats': {
            'U_mean': np.mean(ortho_errors_U),
            'U_std': np.std(ortho_errors_U),
            'V_mean': np.mean(ortho_errors_V),
            'V_std': np.std(ortho_errors_V)
        },
        'descent_violations': descent_violations
    }

    if validation_data is not None:
        analysis_results['validation'] = {
            'ae_mean': np.mean(val_ae_scores),
            'ae_std': np.std(val_ae_scores),
            'detailed_stats': val_detailed_stats
        }

    analysis_file = f"analysis_round{round_idx}_scene{scene_idx}.npz"
    np.savez(analysis_file, **analysis_results)
    print(f"Analysis results saved to {analysis_file}")

    return U_out_all, S_out_all, V_out_all, analysis_results


def test_all_scenes_comprehensive(round_idx=1):
    """全面测试所有场景"""
    scenes = [1, 2, 3]
    all_results = {}

    print("=" * 60)
    print(f"Comprehensive Testing All Scenes for Round {round_idx}")
    print("=" * 60)

    for scene_idx in scenes:
        print(f"\n{'=' * 20} Testing Scene {scene_idx} {'=' * 20}")

        try:
            U_out, S_out, V_out, analysis = test_model_comprehensive(round_idx, scene_idx, use_validation=True)

            all_results[scene_idx] = {
                'outputs': (U_out, S_out, V_out),
                'analysis': analysis
            }

            print(f"✓ Scene {scene_idx} completed successfully")

        except Exception as e:
            print(f"✗ Error testing scene {scene_idx}: {e}")
            continue

    # 生成总结报告
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TESTING SUMMARY")
    print("=" * 60)

    for scene_idx, result in all_results.items():
        analysis = result['analysis']
        print(f"\nScene {scene_idx}:")
        print(f"  Model Parameters: {analysis['complexity']['total_params']:,}")
        print(f"  Estimated FLOPs: {analysis['complexity']['estimated_flops']:,}")
        print(f"  Singular Values - Mean: {analysis['singular_stats']['mean']:.4f}")
        print(f"  Orthogonality Error - U: {analysis['orthogonality_stats']['U_mean']:.4f}")
        print(f"  Orthogonality Error - V: {analysis['orthogonality_stats']['V_mean']:.4f}")
        print(f"  Descent Violations: {analysis['descent_violations']}")

        if 'validation' in analysis:
            print(f"  Validation AE: {analysis['validation']['ae_mean']:.4f} ± {analysis['validation']['ae_std']:.4f}")

    return all_results


def create_advanced_submission_package(round_idx=1):
    """创建高级提交包"""
    print("Creating advanced submission package...")

    # 检查所有必要文件
    required_files = []
    optional_files = []

    # 检查输出文件
    scenes = [1, 2, 3]
    for scene_idx in scenes:
        output_file = f"Round{round_idx}TestOutput{scene_idx}.npz"
        if os.path.exists(output_file):
            required_files.append(output_file)
            print(f"✓ Found output file: {output_file}")
        else:
            print(f"✗ Missing output file: {output_file}")

    # 检查核心文件
    core_files = ["solution.py"]
    for file in core_files:
        if os.path.exists(file):
            required_files.append(file)
            print(f"✓ Found core file: {file}")
        else:
            print(f"✗ Missing core file: {file}")

    # 检查模型权重文件
    for scene_idx in scenes:
        model_file = f"svd_model_round{round_idx}_scene{scene_idx}.pth"
        if os.path.exists(model_file):
            required_files.append(model_file)
            print(f"✓ Found model file: {model_file}")
        else:
            print(f"⚠ Missing model file: {model_file}")

    # 检查分析文件（可选）
    for scene_idx in scenes:
        analysis_file = f"analysis_round{round_idx}_scene{scene_idx}.npz"
        if os.path.exists(analysis_file):
            optional_files.append(analysis_file)

    # 检查训练统计文件（可选）
    for scene_idx in scenes:
        stats_file = f"training_stats_round{round_idx}_scene{scene_idx}.npz"
        if os.path.exists(stats_file):
            optional_files.append(stats_file)

    # 验证输出文件格式
    print("\nValidating output files...")
    for scene_idx in scenes:
        output_file = f"Round{round_idx}TestOutput{scene_idx}.npz"
        if os.path.exists(output_file):
            try:
                data = np.load(output_file)
                required_keys = ['U_out', 'S_out', 'V_out']

                if all(key in data for key in required_keys):
                    print(f"✓ {output_file} format valid")
                    print(f"  U_out: {data['U_out'].shape}, dtype: {data['U_out'].dtype}")
                    print(f"  S_out: {data['S_out'].shape}, dtype: {data['S_out'].dtype}")
                    print(f"  V_out: {data['V_out'].shape}, dtype: {data['V_out'].dtype}")
                else:
                    print(f"✗ {output_file} missing required keys")

            except Exception as e:
                print(f"✗ Error validating {output_file}: {e}")

    # 创建提交包
    if len(required_files) >= len(scenes) + 1:  # 至少要有输出文件和solution.py
        import zipfile

        submission_name = f"advanced_submission_round{round_idx}.zip"

        with zipfile.ZipFile(submission_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 添加必要文件
            for file in required_files:
                zipf.write(file)
                print(f"Added to submission: {file}")

            # 添加可选文件
            for file in optional_files:
                zipf.write(file)
                print(f"Added to submission (optional): {file}")

            # 创建README文件
            readme_content = f"""# SVD Neural Network Submission - Round {round_idx}

## Files Included:
### Required Files:
{chr(10).join(f"- {file}" for file in required_files)}

### Optional Files:
{chr(10).join(f"- {file}" for file in optional_files)}

## Model Architecture:
- Advanced CNN-based feature extraction
- Complex-valued processing
- Gram-Schmidt orthogonalization
- Attention mechanisms

## Training Details:
- Warmup + Cosine annealing learning rate
- Data augmentation
- Early stopping
- Advanced loss function with multiple constraints

## Performance Notes:
- See analysis files for detailed performance metrics
- Model parameters and FLOPs are included in analysis results
"""

            zipf.writestr("README.md", readme_content)
            print("Added README.md to submission")

        print(f"\n✓ Submission package created: {submission_name}")
        print(f"Total files: {len(required_files) + len(optional_files) + 1}")
        return True

    else:
        print("\n✗ Missing required files, cannot create submission package")
        return False


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Advanced SVD Model Testing')
    parser.add_argument('--round', type=int, default=1, help='Round number')
    parser.add_argument('--scene', type=int, default=None, help='Scene number (if None, test all scenes)')
    parser.add_argument('--validation', action='store_true', help='Enable validation mode')
    parser.add_argument('--submit', action='store_true', help='Create submission package')

    args = parser.parse_args()

    if args.scene is not None:
        # 测试单个场景
        test_model_comprehensive(args.round, args.scene, args.validation)
    else:
        # 测试所有场景
        test_all_scenes_comprehensive(args.round)

        if args.submit:
            create_advanced_submission_package(args.round)


if __name__ == "__main__":
    main()