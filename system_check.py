#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import time
import platform
from datetime import datetime


def check_basic_info():
    """检查基础信息"""
    print("🔧 === 基础环境检查 ===")
    print(f"Python版本: {platform.python_version()}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"PyTorch版本: {torch.__version__}")


def check_gpu():
    """检查GPU"""
    print("\n🚀 === GPU检查 ===")

    if torch.cuda.is_available():
        print("✅ CUDA 可用!")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  显存: {props.total_memory / 1024 ** 3:.1f} GB")

            # 清理显存并检查使用情况
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
            print(f"  当前使用: {allocated:.2f} GB")

        return True
    else:
        print("❌ CUDA 不可用")
        print("💡 将使用CPU进行训练")
        return False


def check_memory():
    """检查内存（简化版）"""
    print("\n💾 === 内存检查 ===")

    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"总内存: {memory.total / 1024 ** 3:.1f} GB")
        print(f"可用内存: {memory.available / 1024 ** 3:.1f} GB")
        print(f"使用率: {memory.percent:.1f}%")

        if memory.available / 1024 ** 3 < 4:
            print("⚠️  可用内存较少，建议关闭其他程序")
        else:
            print("✅ 内存充足")

    except ImportError:
        print("⚠️  psutil未安装，无法检查内存")
    except Exception as e:
        print(f"⚠️  内存检查失败: {e}")


def test_model_basic():
    """基础模型测试"""
    print("\n🧪 === 模型测试 ===")

    try:
        from solution import OriginalSVDNet

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")

        # 创建模型
        model = OriginalSVDNet(dim=64, rank=32).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数: {total_params:,}")

        # 测试推理
        dummy_input = torch.randn(64, 64, 2, device=device)

        # 预热
        for _ in range(3):
            with torch.no_grad():
                _ = model(dummy_input)

        # 计时
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()
        num_runs = 10

        for _ in range(num_runs):
            with torch.no_grad():
                U, S, V = model(dummy_input)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs

        print(f"✅ 推理测试成功")
        print(f"平均推理时间: {avg_time * 1000:.2f} ms")
        print(f"输出形状: U{list(U.shape)}, S{list(S.shape)}, V{list(V.shape)}")

        # 显存使用（如果是GPU）
        if device.type == 'cuda':
            memory_used = torch.cuda.max_memory_allocated(device) / 1024 ** 2
            print(f"峰值显存: {memory_used:.1f} MB")

        return True

    except ImportError:
        print("❌ 无法导入solution.py - 请确保文件存在")
        return False
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False


def estimate_training():
    """估算训练时间"""
    print("\n⏱️  === 训练时间估算 ===")

    has_gpu = torch.cuda.is_available()

    if has_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")

        # 简单估算
        if any(x in gpu_name.upper() for x in ['RTX 40', 'RTX 30', 'V100', 'A100']):
            estimated_hours = 1.5
            print("🚀 高性能GPU - 预计训练时间: 1-2小时")
        elif any(x in gpu_name.upper() for x in ['RTX 20', 'GTX 16', 'GTX 10']):
            estimated_hours = 3
            print("⚡ 中等性能GPU - 预计训练时间: 2-4小时")
        else:
            estimated_hours = 6
            print("🐌 入门级GPU - 预计训练时间: 4-8小时")
    else:
        print("🐌 CPU训练 - 预计训练时间: 8-24小时")
        print("💡 强烈建议使用GPU以获得更快速度")


def check_data_files():
    """检查数据文件"""
    print("\n📂 === 数据文件检查 ===")

    import os

    required_files = [
        "DebugData/Round0CfgData1.txt",
        "CompetitionData1/Round1CfgData1.txt",
        "CompetitionData1/Round1TrainData1.npy",
        "CompetitionData1/Round1TrainLabel1.npy"
    ]

    missing_files = []
    existing_files = []

    for file_path in required_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / 1024 / 1024
            print(f"✅ {file_path} ({size_mb:.1f} MB)")
            existing_files.append(file_path)
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)

    if missing_files:
        print(f"\n⚠️  缺少 {len(missing_files)} 个文件")
        print("💡 请确保数据文件在正确位置")
    else:
        print(f"\n✅ 所有数据文件都存在!")

    return len(missing_files) == 0


def main():
    """主检测流程"""
    print("🔍 === 简化系统检测 ===")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 40)

    success_count = 0
    total_checks = 5

    try:
        # 基础检查
        check_basic_info()
        success_count += 1

        # GPU检查
        if check_gpu():
            success_count += 1

        # 内存检查
        check_memory()
        success_count += 1

        # 模型测试
        if test_model_basic():
            success_count += 1

        # 数据文件检查
        if check_data_files():
            success_count += 1

        # 训练估算
        estimate_training()

        print("\n" + "=" * 40)
        print(f"检测完成: {success_count}/{total_checks} 项通过")

        if success_count >= 4:
            print("🎉 系统状态良好，可以开始训练!")
            print("💡 运行: python simple_train.py")
        elif success_count >= 2:
            print("⚠️  系统基本可用，但可能有性能问题")
            print("💡 可以尝试运行训练，但建议解决问题")
        else:
            print("❌ 系统存在问题，建议先解决再训练")

    except Exception as e:
        print(f"\n❌ 检测过程出错: {e}")
        print("💡 请检查Python环境")


if __name__ == "__main__":
    main()