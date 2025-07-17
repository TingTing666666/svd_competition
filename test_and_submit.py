import os
import numpy as np
import torch
from solution import SVDNet, compute_ae_metric
from tqdm import tqdm


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


def test_model(round_idx=1, scene_idx=1):
    """测试单个场景的模型"""

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据路径
    data_dir = f"./CompetitionData{round_idx}"
    cfg_path = f"{data_dir}/Round{round_idx}CfgData{scene_idx}.txt"
    test_data_path = f"{data_dir}/Round{round_idx}TestData{scene_idx}.npy"

    # 读取配置
    samp_num, M, N, IQ, R = read_cfg_file(cfg_path)
    print(f"Testing Scene {scene_idx}: samp_num={samp_num}, M={M}, N={N}, R={R}")

    # 创建模型
    model = SVDNet(M=M, N=N, R=R).to(device)

    # 加载训练好的模型
    model_path = f"svd_model_round{round_idx}_scene{scene_idx}.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found, using random weights")

    # 加载测试数据
    test_data = np.load(test_data_path)  # [N_samp, M, N, 2]
    actual_samp_num = test_data.shape[0]  # 实际样本数
    print(f"Test data shape: {test_data.shape}")
    print(f"Config samp_num: {samp_num}, Actual samp_num: {actual_samp_num}")

    # 使用实际的样本数
    samp_num = actual_samp_num

    # 准备输出数组
    U_out_all = np.zeros((samp_num, M, R, IQ), dtype=np.float32)
    S_out_all = np.zeros((samp_num, R), dtype=np.float32)
    V_out_all = np.zeros((samp_num, N, R, IQ), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        print("Processing test samples...")
        for samp_idx in tqdm(range(samp_num)):
            # 获取单个样本
            H_data = torch.FloatTensor(test_data[samp_idx]).to(device)  # [M, N, 2]

            # 模型推理
            U_out, S_out, V_out = model(H_data)

            # 转换为numpy并保存
            U_out_all[samp_idx] = U_out.cpu().numpy()
            S_out_all[samp_idx] = S_out.cpu().numpy()
            V_out_all[samp_idx] = V_out.cpu().numpy()

    # 保存结果
    output_file = f"Round{round_idx}TestOutput{scene_idx}.npz"
    np.savez(output_file, U_out=U_out_all, S_out=S_out_all, V_out=V_out_all)
    print(f"Results saved to {output_file}")

    return U_out_all, S_out_all, V_out_all


def validate_output_format(U_out, S_out, V_out, expected_shape):
    """验证输出格式"""
    samp_num, M, N, R = expected_shape

    print("Validating output format...")

    # 检查U的形状
    if U_out.shape != (samp_num, M, R, 2):
        print(f"ERROR: U shape mismatch. Expected {(samp_num, M, R, 2)}, got {U_out.shape}")
        return False

    # 检查S的形状
    if S_out.shape != (samp_num, R):
        print(f"ERROR: S shape mismatch. Expected {(samp_num, R)}, got {S_out.shape}")
        return False

    # 检查V的形状
    if V_out.shape != (samp_num, N, R, 2):
        print(f"ERROR: V shape mismatch. Expected {(samp_num, N, R, 2)}, got {V_out.shape}")
        return False

    # 检查奇异值是否为正且降序排列
    for i in range(samp_num):
        if np.any(S_out[i] < 0):
            print(f"ERROR: Negative singular values found in sample {i}")
            return False

        if not np.all(S_out[i][:-1] >= S_out[i][1:]):
            print(f"WARNING: Singular values not in descending order in sample {i}")

    # 检查数据类型
    if U_out.dtype != np.float32:
        print(f"WARNING: U dtype is {U_out.dtype}, expected float32")
    if S_out.dtype != np.float32:
        print(f"WARNING: S dtype is {S_out.dtype}, expected float32")
    if V_out.dtype != np.float32:
        print(f"WARNING: V dtype is {V_out.dtype}, expected float32")

    print("Output format validation passed!")
    return True


def compute_orthogonality_error(matrix):
    """计算正交性误差"""
    # matrix: [M, R, 2] 实虚部格式
    M, R, _ = matrix.shape

    # 转换为复数
    matrix_complex = matrix[..., 0] + 1j * matrix[..., 1]  # [M, R]

    # 计算 U^H * U 或 V^H * V
    gram_matrix = np.conj(matrix_complex.T) @ matrix_complex  # [R, R]

    # 计算与单位矩阵的差异
    identity = np.eye(R)
    error = np.linalg.norm(gram_matrix - identity, 'fro')

    return error


def evaluate_reconstruction_quality(U_out, S_out, V_out, test_data=None):
    """评估重构质量"""
    print("Evaluating reconstruction quality...")

    samp_num = U_out.shape[0]

    # 统计指标
    singular_value_stats = {
        'mean': np.mean(S_out),
        'std': np.std(S_out),
        'min': np.min(S_out),
        'max': np.max(S_out)
    }

    # 正交性误差
    U_ortho_errors = []
    V_ortho_errors = []

    for i in range(min(100, samp_num)):  # 只检查前100个样本以节省时间
        U_ortho_error = compute_orthogonality_error(U_out[i])
        V_ortho_error = compute_orthogonality_error(V_out[i])

        U_ortho_errors.append(U_ortho_error)
        V_ortho_errors.append(V_ortho_error)

    U_ortho_stats = {
        'mean': np.mean(U_ortho_errors),
        'std': np.std(U_ortho_errors),
        'max': np.max(U_ortho_errors)
    }

    V_ortho_stats = {
        'mean': np.mean(V_ortho_errors),
        'std': np.std(V_ortho_errors),
        'max': np.max(V_ortho_errors)
    }

    print(f"Singular value statistics: {singular_value_stats}")
    print(f"U orthogonality error: {U_ortho_stats}")
    print(f"V orthogonality error: {V_ortho_stats}")

    return {
        'singular_value_stats': singular_value_stats,
        'U_ortho_stats': U_ortho_stats,
        'V_ortho_stats': V_ortho_stats
    }


def test_all_scenes(round_idx=1):
    """测试所有场景"""
    scenes = [1, 2, 3]
    all_results = {}

    print("=" * 60)
    print(f"Testing All Scenes for Round {round_idx}")
    print("=" * 60)

    for scene_idx in scenes:
        print(f"\n{'=' * 20} Testing Scene {scene_idx} {'=' * 20}")

        try:
            # 读取配置
            data_dir = f"./CompetitionData{round_idx}"
            cfg_path = f"{data_dir}/Round{round_idx}CfgData{scene_idx}.txt"
            samp_num, M, N, IQ, R = read_cfg_file(cfg_path)

            # 测试模型
            U_out, S_out, V_out = test_model(round_idx, scene_idx)

            # 验证输出格式
            if not validate_output_format(U_out, S_out, V_out, (samp_num, M, N, R)):
                print(f"Scene {scene_idx} failed format validation")
                continue

            # 评估质量
            quality_stats = evaluate_reconstruction_quality(U_out, S_out, V_out)

            all_results[scene_idx] = {
                'U_out': U_out,
                'S_out': S_out,
                'V_out': V_out,
                'quality_stats': quality_stats
            }

            print(f"Scene {scene_idx} completed successfully")

        except Exception as e:
            print(f"Error testing scene {scene_idx}: {e}")
            continue

    print("\n" + "=" * 60)
    print("Testing Summary:")
    print("=" * 60)

    for scene_idx, result in all_results.items():
        stats = result['quality_stats']
        print(f"Scene {scene_idx}:")
        print(f"  - Singular values: mean={stats['singular_value_stats']['mean']:.4f}")
        print(f"  - U orthogonality error: {stats['U_ortho_stats']['mean']:.4f}")
        print(f"  - V orthogonality error: {stats['V_ortho_stats']['mean']:.4f}")

    return all_results


def create_submission_package(round_idx=1):
    """创建提交包"""
    print("Creating submission package...")

    # 检查所有必要文件
    required_files = []

    # 检查输出文件
    scenes = [1, 2, 3]
    for scene_idx in scenes:
        output_file = f"Round{round_idx}TestOutput{scene_idx}.npz"
        if os.path.exists(output_file):
            required_files.append(output_file)
        else:
            print(f"Warning: Missing output file {output_file}")

    # 检查模型文件
    solution_file = "solution.py"
    if os.path.exists(solution_file):
        required_files.append(solution_file)
    else:
        print(f"Error: Missing {solution_file}")
        return False

    # 检查模型权重文件
    for scene_idx in scenes:
        model_file = f"svd_model_round{round_idx}_scene{scene_idx}.pth"
        if os.path.exists(model_file):
            required_files.append(model_file)
        else:
            print(f"Warning: Missing model file {model_file}")

    # 创建压缩包
    import zipfile

    submission_name = f"submission_round{round_idx}.zip"
    with zipfile.ZipFile(submission_name, 'w') as zipf:
        for file in required_files:
            zipf.write(file)
            print(f"Added {file} to submission package")

    print(f"Submission package created: {submission_name}")
    print(f"Total files: {len(required_files)}")

    return True


def debug_single_sample():
    """调试单个样本"""
    print("Debug mode: Testing single sample...")

    # 使用DebugData进行测试
    debug_data_path = "./DebugData/Round0TestData1.npy"
    cfg_path = "./DebugData/Round0CfgData1.txt"

    if not os.path.exists(debug_data_path):
        print(f"Debug data not found: {debug_data_path}")
        return

    # 读取配置
    samp_num, M, N, IQ, R = read_cfg_file(cfg_path)
    print(f"Debug config: M={M}, N={N}, R={R}")

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SVDNet(M=M, N=N, R=R).to(device)

    # 加载测试数据
    test_data = np.load(debug_data_path)
    print(f"Debug data shape: {test_data.shape}")

    # 测试单个样本
    sample_idx = 0
    H_data = torch.FloatTensor(test_data[sample_idx]).to(device)

    model.eval()
    with torch.no_grad():
        U_out, S_out, V_out = model(H_data)

    print(f"Output shapes:")
    print(f"  U: {U_out.shape}")
    print(f"  S: {S_out.shape}")
    print(f"  V: {V_out.shape}")

    print(f"Singular values: {S_out.cpu().numpy()}")

    # 计算正交性
    U_ortho_error = compute_orthogonality_error(U_out.cpu().numpy())
    V_ortho_error = compute_orthogonality_error(V_out.cpu().numpy())

    print(f"U orthogonality error: {U_ortho_error:.4f}")
    print(f"V orthogonality error: {V_ortho_error:.4f}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Test SVD Neural Network')
    parser.add_argument('--round', type=int, default=1, help='Round number')
    parser.add_argument('--scene', type=int, default=None, help='Scene number (if None, test all scenes)')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--submit', action='store_true', help='Create submission package')

    args = parser.parse_args()

    if args.debug:
        debug_single_sample()
    elif args.scene is not None:
        # 测试单个场景
        test_model(args.round, args.scene)
    else:
        # 测试所有场景
        test_all_scenes(args.round)

        if args.submit:
            create_submission_package(args.round)


if __name__ == "__main__":
    main()