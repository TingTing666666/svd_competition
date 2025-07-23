import os
import numpy as np
import torch
from solution import SVDNet, compute_ae_metric
from tqdm import tqdm


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    data_dir = f"./CompetitionData{round_idx}"
    cfg_path = f"{data_dir}/Round{round_idx}CfgData{scene_idx}.txt"
    test_data_path = f"{data_dir}/Round{round_idx}TestData{scene_idx}.npy"

    # Load config
    samp_num, M, N, IQ, R = read_cfg_file(cfg_path)

    # Load model
    model = SVDNet(M=M, N=N, R=R).to(device)
    model_path = f"svd_model_round{round_idx}_scene{scene_idx}.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    # Load test data
    test_data = np.load(test_data_path)
    actual_samp_num = test_data.shape[0]

    # Format validation
    assert test_data.shape[1:] == (
    M, N, IQ), f"Data shape mismatch: expected ({M}, {N}, {IQ}), got {test_data.shape[1:]}"
    assert IQ == 2, f"IQ must be 2, got {IQ}"

    # Prepare outputs
    U_out_all = np.zeros((actual_samp_num, M, R, IQ), dtype=np.float32)
    S_out_all = np.zeros((actual_samp_num, R), dtype=np.float32)
    V_out_all = np.zeros((actual_samp_num, N, R, IQ), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for samp_idx in tqdm(range(actual_samp_num), desc="Processing"):
            H_data = torch.FloatTensor(test_data[samp_idx]).to(device)

            # Validate input shape
            assert H_data.shape == (M, N, IQ), f"Input shape error: expected ({M}, {N}, {IQ}), got {H_data.shape}"

            U_out, S_out, V_out = model(H_data)

            # Validate output shapes
            assert U_out.shape == (M, R, IQ), f"U output shape error: expected ({M}, {R}, {IQ}), got {U_out.shape}"
            assert S_out.shape == (R,), f"S output shape error: expected ({R},), got {S_out.shape}"
            assert V_out.shape == (N, R, IQ), f"V output shape error: expected ({N}, {R}, {IQ}), got {V_out.shape}"

            # Validate singular values
            S_numpy = S_out.cpu().numpy()
            assert np.all(S_numpy >= 0), f"Negative singular values found in sample {samp_idx}"

            U_out_all[samp_idx] = U_out.cpu().numpy().astype(np.float32)
            S_out_all[samp_idx] = S_out.cpu().numpy().astype(np.float32)
            V_out_all[samp_idx] = V_out.cpu().numpy().astype(np.float32)

    # Save output
    output_file = f"Round{round_idx}TestOutput{scene_idx}.npz"
    np.savez(output_file, U_out=U_out_all, S_out=S_out_all, V_out=V_out_all)

    # Validate saved file
    loaded = np.load(output_file)
    assert 'U_out' in loaded and 'S_out' in loaded and 'V_out' in loaded, "Missing keys in output file"

    print(f"Scene {scene_idx} completed. Output: {output_file}")
    return U_out_all, S_out_all, V_out_all


def test_all_scenes(round_idx=1):
    scenes = [1, 2, 3]
    for scene_idx in scenes:
        try:
            test_model(round_idx, scene_idx)
        except Exception as e:
            print(f"Error testing scene {scene_idx}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=int, default=None)
    parser.add_argument('--round', type=int, default=1)
    args = parser.parse_args()

    if args.scene is not None:
        test_model(args.round, args.scene)
    else:
        test_all_scenes(args.round)