import os
import numpy as np
import torch
from solution import SVDNet, compute_ae_metric
from tqdm import tqdm
import zipfile


def read_cfg_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    samp_num = int(lines[0].strip())
    M = int(lines[1].strip())
    N = int(lines[2].strip())
    IQ = int(lines[3].strip())
    R = int(lines[4].strip())
    return samp_num, M, N, IQ, R


def test_model(round_idx=1, scene_idx=1):
    print(f"Testing Scene {scene_idx} (Round {round_idx})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 数据路径
    data_dir = f"./CompetitionData{round_idx}"
    cfg_path = f"{data_dir}/Round{round_idx}CfgData{scene_idx}.txt"
    test_data_path = f"{data_dir}/Round{round_idx}TestData{scene_idx}.npy"

    # 读取配置
    samp_num, M, N, IQ, R = read_cfg_file(cfg_path)
    print(f"Config: samp_num={samp_num}, M={M}, N={N}, R={R}")

    # 创建模型
    model = SVDNet(M=M, N=N, R=R).to(device)

    # 加载训练好的模型
    model_path = f"svd_model_round{round_idx}_scene{scene_idx}.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded: {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found")

    # 加载测试数据
    test_data = np.load(test_data_path)
    actual_samp_num = test_data.shape[0]
    print(f"Test data shape: {test_data.shape}")

    # 使用实际样本数
    samp_num = actual_samp_num

    # 准备输出数组
    U_out_all = np.zeros((samp_num, M, R, IQ), dtype=np.float32)
    S_out_all = np.zeros((samp_num, R), dtype=np.float32)
    V_out_all = np.zeros((samp_num, N, R, IQ), dtype=np.float32)

    # 推理设置
    model.eval()
    torch.set_grad_enabled(False)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("Processing test samples...")
    for samp_idx in tqdm(range(samp_num), desc="Testing"):
        H_data = torch.FloatTensor(test_data[samp_idx]).to(device, non_blocking=True)

        # 模型推理
        U_out, S_out, V_out = model(H_data)

        # 转换并保存
        U_out_all[samp_idx] = U_out.cpu().numpy().astype(np.float32)
        S_out_all[samp_idx] = S_out.cpu().numpy().astype(np.float32)
        V_out_all[samp_idx] = V_out.cpu().numpy().astype(np.float32)

    # 验证输出
    assert U_out_all.shape == (samp_num, M, R, IQ), f"U shape error: {U_out_all.shape}"
    assert S_out_all.shape == (samp_num, R), f"S shape error: {S_out_all.shape}"
    assert V_out_all.shape == (samp_num, N, R, IQ), f"V shape error: {V_out_all.shape}"
    assert np.all(S_out_all >= 0), "Negative singular values detected"

    # 保存结果
    output_file = f"Round{round_idx}TestOutput{scene_idx}.npz"
    np.savez_compressed(output_file, U_out=U_out_all, S_out=S_out_all, V_out=V_out_all)
    print(f"Results saved: {output_file}")

    return U_out_all, S_out_all, V_out_all


def compute_orthogonality_error(matrix):
    M, R, _ = matrix.shape
    matrix_complex = matrix[..., 0] + 1j * matrix[..., 1]
    gram_matrix = np.conj(matrix_complex.T) @ matrix_complex
    identity = np.eye(R)
    error = np.linalg.norm(gram_matrix - identity, 'fro')
    return error


def evaluate_quality(U_out, S_out, V_out):
    samp_num = U_out.shape[0]

    # 奇异值统计
    sv_mean = np.mean(S_out)
    sv_std = np.std(S_out)
    sv_min = np.min(S_out)
    sv_max = np.max(S_out)

    # 正交性评估（采样）
    sample_size = min(100, samp_num)
    sample_indices = np.random.choice(samp_num, sample_size, replace=False)

    U_ortho_errors = []
    V_ortho_errors = []

    for idx in sample_indices:
        U_ortho_error = compute_orthogonality_error(U_out[idx])
        V_ortho_error = compute_orthogonality_error(V_out[idx])
        U_ortho_errors.append(U_ortho_error)
        V_ortho_errors.append(V_ortho_error)

    U_ortho_mean = np.mean(U_ortho_errors)
    V_ortho_mean = np.mean(V_ortho_errors)

    # 预估AE分数
    estimated_ae = U_ortho_mean + V_ortho_mean + 0.05

    print(f"Singular values: [{sv_min:.4f}, {sv_max:.4f}], mean={sv_mean:.4f}")
    print(f"U orthogonality error: {U_ortho_mean:.4f}")
    print(f"V orthogonality error: {V_ortho_mean:.4f}")
    print(f"Estimated AE: {estimated_ae:.4f}")

    return estimated_ae


def test_all_scenes(round_idx=1):
    scenes = [1, 2, 3]
    results = {}

    print(f"Testing All Scenes for Round {round_idx}")
    print("=" * 50)

    for scene_idx in scenes:
        print(f"\nTesting Scene {scene_idx}")
        print("-" * 30)

        try:
            U_out, S_out, V_out = test_model(round_idx, scene_idx)
            estimated_ae = evaluate_quality(U_out, S_out, V_out)

            results[scene_idx] = {
                'status': 'success',
                'estimated_ae': estimated_ae
            }

            print(f"Scene {scene_idx} completed successfully")

        except Exception as e:
            print(f"Error testing scene {scene_idx}: {e}")
            results[scene_idx] = {
                'status': 'failed',
                'error': str(e)
            }
            continue

    # 总结
    print(f"\nTesting Summary:")
    print("=" * 30)

    successful_scenes = [s for s, r in results.items() if r['status'] == 'success']

    if successful_scenes:
        total_ae = sum(results[s]['estimated_ae'] for s in successful_scenes)
        avg_ae = total_ae / len(successful_scenes)

        print(f"Successful scenes: {successful_scenes}")
        print(f"Average estimated AE: {avg_ae:.4f}")

        for scene_idx in successful_scenes:
            ae = results[scene_idx]['estimated_ae']
            print(f"  Scene {scene_idx}: {ae:.4f}")
    else:
        print("No scenes completed successfully")

    return results


def create_submission_package(round_idx=1):
    print("Creating submission package...")

    required_files = []
    scenes = [1, 2, 3]

    # 检查输出文件
    for scene_idx in scenes:
        output_file = f"Round{round_idx}TestOutput{scene_idx}.npz"
        if os.path.exists(output_file):
            required_files.append(output_file)
            print(f"Found: {output_file}")
        else:
            print(f"Missing: {output_file}")

    # 检查solution.py
    if os.path.exists("solution.py"):
        required_files.append("solution.py")
        print(f"Found: solution.py")
    else:
        print(f"Missing: solution.py")
        return False

    # 检查模型文件
    for scene_idx in scenes:
        model_file = f"svd_model_round{round_idx}_scene{scene_idx}.pth"
        if os.path.exists(model_file):
            required_files.append(model_file)
            print(f"Found: {model_file}")

    if len([f for f in required_files if f.endswith('.npz')]) < 3:
        print("Warning: Not all output files found")

    # 创建zip文件
    submission_name = f"submission_round{round_idx}.zip"
    with zipfile.ZipFile(submission_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in required_files:
            zipf.write(file)
            print(f"Added: {file}")

    file_size = os.path.getsize(submission_name) / 1024 / 1024
    print(f"Submission package created: {submission_name} ({file_size:.1f} MB)")
    print(f"Total files: {len(required_files)}")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test SVD Neural Network')
    parser.add_argument('--round', type=int, default=1, help='Round number')
    parser.add_argument('--scene', type=int, default=None, help='Scene number (if None, test all scenes)')
    parser.add_argument('--submit', action='store_true', help='Create submission package')

    args = parser.parse_args()

    if args.scene is not None:
        test_model(args.round, args.scene)
    else:
        test_all_scenes(args.round)
        if args.submit:
            create_submission_package(args.round)