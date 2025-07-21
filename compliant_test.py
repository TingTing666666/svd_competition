import os
import numpy as np
import torch
from solution import SVDNet, compute_ae_metric
from tqdm import tqdm
import zipfile
from datetime import datetime


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


def test_model_compliant(round_idx=1, scene_idx=1):
    """符合要求的测试函数"""

    print(f"🧪 Testing Scene {scene_idx} (Round {round_idx})")

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")

    # 数据路径
    data_dir = f"./CompetitionData{round_idx}"
    cfg_path = f"{data_dir}/Round{round_idx}CfgData{scene_idx}.txt"
    test_data_path = f"{data_dir}/Round{round_idx}TestData{scene_idx}.npy"

    # 读取配置
    samp_num, M, N, IQ, R = read_cfg_file(cfg_path)
    print(f"📊 Config: samp_num={samp_num}, M={M}, N={N}, R={R}")

    # 创建模型
    model = SVDNet(M=M, N=N, R=R).to(device)

    # 加载模型权重
    model_path = f"svd_model_round{round_idx}_scene{scene_idx}.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"📦 Model loaded: {model_path}")
    else:
        print(f"⚠️ Model file not found: {model_path}, using random weights")

    # 加载测试数据
    test_data = np.load(test_data_path)
    actual_samp_num = test_data.shape[0]
    print(f"📈 Test data shape: {test_data.shape}")

    # 使用实际的样本数
    samp_num = actual_samp_num

    # 准备输出数组（严格按照比赛要求的格式）
    U_out_all = np.zeros((samp_num, M, R, IQ), dtype=np.float32)
    S_out_all = np.zeros((samp_num, R), dtype=np.float32)
    V_out_all = np.zeros((samp_num, N, R, IQ), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        print("🔄 Processing test samples...")

        for samp_idx in tqdm(range(samp_num), desc="Testing"):
            # 获取单个样本
            H_data = torch.FloatTensor(test_data[samp_idx]).to(device)  # [M, N, 2]

            # 模型推理
            U_out, S_out, V_out = model(H_data)

            # 转换为numpy并保存（确保数据类型正确）
            U_out_all[samp_idx] = U_out.cpu().numpy().astype(np.float32)
            S_out_all[samp_idx] = S_out.cpu().numpy().astype(np.float32)
            V_out_all[samp_idx] = V_out.cpu().numpy().astype(np.float32)

    # 验证输出格式
    print("🔍 Validating output format...")
    assert U_out_all.shape == (samp_num, M, R, IQ), f"U shape error: {U_out_all.shape}"
    assert S_out_all.shape == (samp_num, R), f"S shape error: {S_out_all.shape}"
    assert V_out_all.shape == (samp_num, N, R, IQ), f"V shape error: {V_out_all.shape}"

    # 检查数据类型
    assert U_out_all.dtype == np.float32, f"U dtype error: {U_out_all.dtype}"
    assert S_out_all.dtype == np.float32, f"S dtype error: {S_out_all.dtype}"
    assert V_out_all.dtype == np.float32, f"V dtype error: {V_out_all.dtype}"

    # 检查奇异值是否为正
    assert np.all(S_out_all >= 0), "Negative singular values detected"

    print("✅ Output format validation passed!")

    # 保存结果（标准格式 - 严格按照比赛要求）
    timestamp = datetime.now().strftime("%y_%m_%d_%H_%M")

    # 带时间戳的文件
    output_file_timestamped = f"Round{round_idx}TestOutput{scene_idx}_{timestamp}.npz"
    np.savez(output_file_timestamped, U_out=U_out_all, S_out=S_out_all, V_out=V_out_all)
    print(f"💾 Results saved: {output_file_timestamped}")

    # 标准命名的文件（用于提交）
    standard_output_file = f"Round{round_idx}TestOutput{scene_idx}.npz"
    np.savez(standard_output_file, U_out=U_out_all, S_out=S_out_all, V_out=V_out_all)
    print(f"📋 Standard copy saved: {standard_output_file}")

    return U_out_all, S_out_all, V_out_all


def quick_quality_check(U_out, S_out, V_out, scene_idx):
    """快速质量检查"""
    print(f"\n📊 Quality Check for Scene {scene_idx}:")

    # 奇异值统计
    print(f"Singular values - Mean: {np.mean(S_out):.4f}, Std: {np.std(S_out):.4f}")
    print(f"Singular values - Range: [{np.min(S_out):.4f}, {np.max(S_out):.4f}]")

    # 检查单调性
    monotonic_violations = 0
    for i in range(min(100, S_out.shape[0])):
        if not np.all(S_out[i][:-1] >= S_out[i][1:] - 1e-6):
            monotonic_violations += 1

    print(f"Monotonic violations: {monotonic_violations}/{min(100, S_out.shape[0])}")

    # 正交性检查（采样检查）
    def check_orthogonality(matrices, sample_size=50):
        sample_indices = np.random.choice(matrices.shape[0], min(sample_size, matrices.shape[0]), replace=False)
        errors = []

        for idx in sample_indices:
            # 转换为复数
            matrix_complex = matrices[idx, :, :, 0] + 1j * matrices[idx, :, :, 1]

            # 计算 A^H * A
            gram = np.conj(matrix_complex.T) @ matrix_complex
            identity = np.eye(gram.shape[0])
            error = np.linalg.norm(gram - identity, 'fro')
            errors.append(error)

        return np.array(errors)

    U_ortho_errors = check_orthogonality(U_out)
    V_ortho_errors = check_orthogonality(V_out)

    print(f"U orthogonality error: {np.mean(U_ortho_errors):.4f} ± {np.std(U_ortho_errors):.4f}")
    print(f"V orthogonality error: {np.mean(V_ortho_errors):.4f} ± {np.std(V_ortho_errors):.4f}")

    # 估算AE分数
    estimated_ae = np.mean(U_ortho_errors) + np.mean(V_ortho_errors) + 0.1  # 估算重构误差
    print(f"🎯 Estimated AE score: {estimated_ae:.4f}")

    return estimated_ae


def test_all_scenes_compliant(round_idx=1):
    """测试所有场景 - 符合要求版本"""
    scenes = [1, 2, 3]
    all_results = {}
    estimated_scores = {}

    print("🏆" * 15)
    print(f"COMPLIANT TESTING - ROUND {round_idx}")
    print("🏆" * 15)

    for scene_idx in scenes:
        print(f"\n{'=' * 40}")
        print(f"🎯 Testing Scene {scene_idx}")
        print(f"{'=' * 40}")

        try:
            # 测试模型
            U_out, S_out, V_out = test_model_compliant(round_idx, scene_idx)

            # 质量检查
            estimated_ae = quick_quality_check(U_out, S_out, V_out, scene_idx)

            all_results[scene_idx] = {
                'U_out': U_out,
                'S_out': S_out,
                'V_out': V_out,
                'status': 'success'
            }
            estimated_scores[scene_idx] = estimated_ae

            print(f"✅ Scene {scene_idx} completed successfully")

        except Exception as e:
            print(f"💥 Error testing scene {scene_idx}: {e}")
            estimated_scores[scene_idx] = float('inf')
            continue

    # 整体总结
    print("\n🏆" + "=" * 40)
    print("TESTING SUMMARY")
    print("=" * 40 + "🏆")

    successful_scenes = [s for s, r in all_results.items() if r.get('status') == 'success']

    if successful_scenes:
        avg_estimated_ae = np.mean([estimated_scores[s] for s in successful_scenes])
        print(f"✅ Successful scenes: {successful_scenes}")
        print(f"🎯 Average estimated AE: {avg_estimated_ae:.4f}")

        # 预测排名
        if avg_estimated_ae < 1.0:
            print("🥇 Target ranking: TOP 5!")
        elif avg_estimated_ae < 1.5:
            print("🥈 Target ranking: TOP 10!")
        elif avg_estimated_ae < 2.5:
            print("🥉 Target ranking: TOP 20!")
        else:
            print("💪 Need more optimization!")

    return all_results


def create_compliant_submission(round_idx=1):
    """创建完全符合要求的提交包"""
    timestamp = datetime.now().strftime("%y_%m_%d_%H_%M")
    print(f"📦 Creating compliant submission package... {timestamp}")

    # 检查必要文件
    required_files = []
    scenes = [1, 2, 3]

    # 1. 检查输出文件
    missing_outputs = []
    for scene_idx in scenes:
        output_file = f"Round{round_idx}TestOutput{scene_idx}.npz"
        if os.path.exists(output_file):
            required_files.append(output_file)
            print(f"✅ Found: {output_file}")
        else:
            missing_outputs.append(output_file)
            print(f"❌ Missing: {output_file}")

    if missing_outputs:
        print(f"⚠️ Missing output files: {missing_outputs}")
        print("Please run testing first!")
        return False

    # 2. 检查solution.py
    solution_file = "solution.py"
    if os.path.exists(solution_file):
        required_files.append(solution_file)
        print(f"✅ Found: {solution_file}")
    else:
        print(f"❌ Missing: {solution_file}")
        return False

    # 3. 检查模型文件（可选）
    optional_files = []
    for scene_idx in scenes:
        model_file = f"svd_model_round{round_idx}_scene{scene_idx}.pth"
        if os.path.exists(model_file):
            optional_files.append(model_file)
            print(f"📦 Found model: {model_file}")

    # 4. 验证输出文件格式
    print("\n🔍 Validating output files...")
    for output_file in [f for f in required_files if f.endswith('.npz')]:
        try:
            data = np.load(output_file)
            assert 'U_out' in data, f"Missing U_out in {output_file}"
            assert 'S_out' in data, f"Missing S_out in {output_file}"
            assert 'V_out' in data, f"Missing V_out in {output_file}"

            U_out = data['U_out']
            S_out = data['S_out']
            V_out = data['V_out']

            # 检查数据类型
            assert U_out.dtype == np.float32, f"Wrong U_out dtype in {output_file}"
            assert S_out.dtype == np.float32, f"Wrong S_out dtype in {output_file}"
            assert V_out.dtype == np.float32, f"Wrong V_out dtype in {output_file}"

            print(f"✅ Validated: {output_file}")

        except Exception as e:
            print(f"❌ Validation failed for {output_file}: {e}")
            return False

    # 5. 创建提交包
    all_files = required_files + optional_files

    # 按照你的要求，ZIP文件名包含时间戳
    submission_name = f"submission_round{round_idx}_{timestamp}.zip"

    try:
        with zipfile.ZipFile(submission_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in all_files:
                if os.path.exists(file):
                    zipf.write(file)
                    print(f"📥 Added: {file}")

        # 获取文件大小
        package_size = os.path.getsize(submission_name) / (1024 * 1024)  # MB

        print(f"\n🏆 SUBMISSION PACKAGE CREATED! 🏆")
        print(f"📦 Package: {submission_name}")
        print(f"📊 Size: {package_size:.2f} MB")
        print(f"📁 Files: {len(all_files)}")
        print(f"🎯 Ready for submission!")

        return submission_name

    except Exception as e:
        print(f"❌ Failed to create submission package: {e}")
        return False


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Compliant SVD Testing & Submission')
    parser.add_argument('--round', type=int, default=1, help='Round number')
    parser.add_argument('--scene', type=int, default=None, help='Scene number (if None, test all scenes)')
    parser.add_argument('--submit', action='store_true', help='Create submission package')
    parser.add_argument('--validate-only', action='store_true', help='Only validate existing results')

    args = parser.parse_args()

    if args.validate_only:
        # 只验证现有结果
        scenes = [1, 2, 3] if args.scene is None else [args.scene]
        for scene_idx in scenes:
            output_file = f"Round{args.round}TestOutput{scene_idx}.npz"
            if os.path.exists(output_file):
                data = np.load(output_file)
                U_out = data['U_out']
                S_out = data['S_out']
                V_out = data['V_out']
                quick_quality_check(U_out, S_out, V_out, scene_idx)
            else:
                print(f"❌ No results found for scene {scene_idx}")

    elif args.scene is not None:
        # 测试单个场景
        test_model_compliant(args.round, args.scene)
    else:
        # 测试所有场景
        test_all_scenes_compliant(args.round)

        if args.submit:
            create_compliant_submission(args.round)


if __name__ == "__main__":
    main()