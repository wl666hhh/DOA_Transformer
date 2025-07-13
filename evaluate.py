import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import config
from model import DOATransformer
from data_generator import generate_ula_data


def predict(model, src_tensor):
    """预测函数，返回预测的角度"""
    model.eval()
    device = config.DEVICE
    src_tensor = src_tensor.to(device)

    # Encoder processing
    src_proj = model.input_proj(src_tensor) * np.sqrt(config.D_MODEL)
    src_pos = model.pos_encoder(src_proj.transpose(0, 1)).transpose(0, 1)
    memory = model.transformer.encoder(src_pos)

    # Decoder processing (autoregressive)
    tgt_tokens = torch.full((1, 1), config.START_TOKEN, dtype=torch.long, device=device)

    for _ in range(config.MAX_SEQ_LENGTH - 1):
        tgt_emb = model.embedding(tgt_tokens) * np.sqrt(config.D_MODEL)
        tgt_pos = model.pos_decoder(tgt_emb.transpose(0, 1)).transpose(0, 1)

        tgt_mask = model.transformer.generate_square_subsequent_mask(tgt_tokens.size(1)).to(device)

        output = model.transformer.decoder(tgt_pos, memory, tgt_mask=tgt_mask)

        pred_prob = model.fc_out(output[:, -1, :])
        next_token = pred_prob.argmax(1)

        if next_token.item() == config.END_TOKEN:
            break

        tgt_tokens = torch.cat([tgt_tokens, next_token.unsqueeze(0)], dim=1)

    # Convert tokens to angles
    pred_indices = tgt_tokens.squeeze(0).cpu().numpy()[1:]  # Exclude START_TOKEN
    valid_indices = pred_indices[pred_indices < config.GRID_SIZE]
    pred_angles = valid_indices * config.ANGLE_RESOLUTION + config.ANGLE_MIN

    return np.sort(pred_angles)


def calculate_hausdorff_distance(true_angles, pred_angles):
    """计算豪斯多夫距离"""
    if len(pred_angles) == 0:
        return float('inf')

    # 计算双向豪斯多夫距离
    h1 = max([min([abs(t - p) for p in pred_angles]) for t in true_angles])
    h2 = max([min([abs(p - t) for t in true_angles]) for p in pred_angles])

    return max(h1, h2)


def run_fixed_angle_evaluation(model, snr_db):
    """运行固定角度评估（表4.2的实验）"""
    print(f"\n--- Fixed Angle Evaluation at SNR={snr_db}dB ---")

    # 根据论文中的固定角度设置
    fixed_scenarios = {
        1: [[-2.2]],  # 1个信源
        2: [[-2.2, 0.1]],  # 2个信源
        3: [[-2.2, 0.1, 1.8]]  # 3个信源
    }

    results = {}

    for num_sources, angle_sets in fixed_scenarios.items():
        hausdorff_distances = []
        max_hausdorff_distances = []
        success_count = 0

        print(f"Testing {num_sources} source(s)...")

        for angles in angle_sets:
            # 运行1000次实验
            for _ in tqdm(range(1000), desc=f"Running {num_sources} sources"):
                R, _ = generate_ula_data(
                    num_sources=len(angles),
                    thetas_deg=angles,
                    snr_db=snr_db,
                    M=config.NUM_ARRAY_ELEMENTS,
                    L=config.SNAPSHOTS,
                    d_lambda=config.ARRAY_SPACING
                )

                real_part = torch.from_numpy(R.real.astype(np.float32))
                imag_part = torch.from_numpy(R.imag.astype(np.float32))
                src_tensor = torch.stack([real_part, imag_part], dim=0).view(2, -1).T.unsqueeze(0)

                pred_angles = predict(model, src_tensor)

                # 计算豪斯多夫距离
                hd = calculate_hausdorff_distance(angles, pred_angles)
                hausdorff_distances.append(hd)
                max_hausdorff_distances.append(hd)  # 对于单次实验，平均和最大是一样的

                # 判断是否成功恢复（豪斯多夫距离小于阈值，如1度）
                if hd < 1.0 and len(pred_angles) == len(angles):
                    success_count += 1

        avg_hd = np.mean(hausdorff_distances)
        max_hd = np.max(max_hausdorff_distances)
        success_rate = success_count / 1000 * 100

        results[num_sources] = {
            'avg_hd': avg_hd,
            'max_hd': max_hd,
            'success_rate': success_rate
        }

        print(f"Sources: {num_sources}, Avg HD: {avg_hd:.2f}, Max HD: {max_hd:.2f}, Success Rate: {success_rate:.1f}%")

    return results


def run_varying_angle_evaluation(model, num_sources, snr_db):
    """运行变化角度评估（图4.8和4.9的实验）"""
    print(f"\n--- Varying Angle Evaluation for {num_sources} source(s) at SNR={snr_db}dB ---")

    true_angles_list = []
    pred_angles_list = []
    errors_list = []

    # 根据论文中的角度范围设置
    if num_sources == 1:
        # 从-39.8到39.2，步长为1
        angle_range = np.arange(-39.8, 39.3, 1.0)
        scenarios = [[a] for a in angle_range]
    elif num_sources == 2:
        # 第一个信号：-39.8到-36.2，第二个信号：-36.8到39.2
        angle_range1 = np.arange(-39.8, -36.1, 1.0)
        angle_range2 = np.arange(-36.8, 39.3, 1.0)
        min_len = min(len(angle_range1), len(angle_range2))
        scenarios = [[angle_range1[i], angle_range2[i]] for i in range(min_len)]
    elif num_sources == 3:
        # 三个信号的角度范围，保持3度间隔
        angle_range1 = np.arange(-39.8, -33.1, 1.0)
        angle_range2 = np.arange(-36.8, -36.1, 1.0)
        angle_range3 = np.arange(-33.8, 39.3, 1.0)
        min_len = min(len(angle_range1), len(angle_range2), len(angle_range3))
        scenarios = [[angle_range1[i], angle_range2[i] if i < len(angle_range2) else angle_range2[-1],
                      angle_range3[i] if i < len(angle_range3) else angle_range3[-1]] for i in range(min_len)]

    for true_thetas in tqdm(scenarios, desc=f"Predicting {num_sources} sources"):
        R, _ = generate_ula_data(
            num_sources=len(true_thetas),
            thetas_deg=true_thetas,
            snr_db=snr_db,
            M=config.NUM_ARRAY_ELEMENTS,
            L=config.SNAPSHOTS,
            d_lambda=config.ARRAY_SPACING
        )

        real_part = torch.from_numpy(R.real.astype(np.float32))
        imag_part = torch.from_numpy(R.imag.astype(np.float32))
        src_tensor = torch.stack([real_part, imag_part], dim=0).view(2, -1).T.unsqueeze(0)

        pred_thetas = predict(model, src_tensor)

        # 存储结果
        for i, true_theta in enumerate(true_thetas):
            true_angles_list.append(true_theta)
            if i < len(pred_thetas):
                pred_angles_list.append(pred_thetas[i])
                errors_list.append(abs(pred_thetas[i] - true_theta))
            else:
                pred_angles_list.append(np.nan)
                errors_list.append(np.nan)

    return true_angles_list, pred_angles_list, errors_list


def plot_varying_angle_results(results_25db, results_35db):
    """绘制变化角度结果图（复制论文中的图4.8和4.9）"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Transformer Model Performance with Varying Angles', fontsize=16)

    # 绘制25dB结果
    for i, (num_sources, (true_angles, pred_angles, errors)) in enumerate(results_25db.items()):
        ax = axes[0, i]

        # 主要的预测vs真实角度图
        ax.plot([min(true_angles), max(true_angles)], [min(true_angles), max(true_angles)], 'r--',
                label='Perfect Prediction', linewidth=2)

        # 过滤掉NaN值
        valid_mask = ~np.isnan(pred_angles)
        ax.scatter(np.array(true_angles)[valid_mask], np.array(pred_angles)[valid_mask],
                   s=15, alpha=0.7, c='blue', label='Prediction')

        ax.set_title(f'{num_sources} Source(s) - SNR=25dB')
        ax.set_xlabel('True Angle θ (degrees)')
        if i == 0:
            ax.set_ylabel('Estimated Angle θ̂ (degrees)')
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend()

        # 设置坐标轴范围
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)

    # 绘制35dB结果
    for i, (num_sources, (true_angles, pred_angles, errors)) in enumerate(results_35db.items()):
        ax = axes[1, i]

        ax.plot([min(true_angles), max(true_angles)], [min(true_angles), max(true_angles)], 'r--',
                label='Perfect Prediction', linewidth=2)

        valid_mask = ~np.isnan(pred_angles)
        ax.scatter(np.array(true_angles)[valid_mask], np.array(pred_angles)[valid_mask],
                   s=15, alpha=0.7, c='blue', label='Prediction')

        ax.set_title(f'{num_sources} Source(s) - SNR=35dB')
        ax.set_xlabel('True Angle θ (degrees)')
        if i == 0:
            ax.set_ylabel('Estimated Angle θ̂ (degrees)')
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend()

        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs("plots", exist_ok=True)
    plt.savefig('plots/transformer_varying_angles.png', dpi=300, bbox_inches='tight')
    print("Varying angle results saved to plots/transformer_varying_angles.png")
    plt.show()


def plot_error_analysis(results_25db, results_35db):
    """绘制误差分析图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Error Analysis: |θ - θ̂|', fontsize=16)

    # 25dB误差分析
    for i, (num_sources, (true_angles, pred_angles, errors)) in enumerate(results_25db.items()):
        ax = axes[0, i]

        valid_errors = [e for e in errors if not np.isnan(e)]
        if valid_errors:
            ax.plot(range(len(valid_errors)), valid_errors, 'b-', alpha=0.7, linewidth=1)
            ax.axhline(y=1.0, color='r', linestyle='--', label='1° threshold')
            ax.set_title(f'{num_sources} Source(s) - SNR=25dB')
            ax.set_xlabel('Sample Index')
            if i == 0:
                ax.set_ylabel('|θ - θ̂| (degrees)')
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend()
            ax.set_ylim(0, max(5, max(valid_errors) * 1.1))

    # 35dB误差分析
    for i, (num_sources, (true_angles, pred_angles, errors)) in enumerate(results_35db.items()):
        ax = axes[1, i]

        valid_errors = [e for e in errors if not np.isnan(e)]
        if valid_errors:
            ax.plot(range(len(valid_errors)), valid_errors, 'b-', alpha=0.7, linewidth=1)
            ax.axhline(y=1.0, color='r', linestyle='--', label='1° threshold')
            ax.set_title(f'{num_sources} Source(s) - SNR=35dB')
            ax.set_xlabel('Sample Index')
            if i == 0:
                ax.set_ylabel('|θ - θ̂| (degrees)')
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend()
            ax.set_ylim(0, max(1.5, max(valid_errors) * 1.1))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('plots/error_analysis.png', dpi=300, bbox_inches='tight')
    print("Error analysis saved to plots/error_analysis.png")
    plt.show()


def print_performance_table(results_25db, results_35db):
    """打印性能表格（表4.2）"""
    print("\n" + "=" * 80)
    print("Performance Table (Table 4.2 Replication)")
    print("=" * 80)
    print(f"{'Source Count':<12} {'SNR=25dB':<25} {'SNR=35dB':<25}")
    print(f"{'':>12} {'H_d':<8} {'max(H_d)':<8} {'Success':<8} {'H_d':<8} {'max(H_d)':<8} {'Success':<8}")
    print("-" * 80)

    for num_sources in [1, 2, 3]:
        r25 = results_25db[num_sources]
        r35 = results_35db[num_sources]
        print(f"{num_sources:<12} {r25['avg_hd']:<8.2f} {r25['max_hd']:<8.2f} {r25['success_rate']:<8.1f}% "
              f"{r35['avg_hd']:<8.2f} {r35['max_hd']:<8.2f} {r35['success_rate']:<8.1f}%")


def main():
    device = config.DEVICE
    model = DOATransformer(
        d_model=config.D_MODEL,
        nhead=config.N_HEAD,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        dim_feedforward=config.D_FF,
        dropout=config.DROPOUT,
        vocab_size=config.VOCAB_SIZE
    ).to(device)

    if not os.path.exists(config.MODEL_PATH):
        print(f"Model not found at {config.MODEL_PATH}. Please train the model first.")
        return

    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))

    # 1. 运行固定角度评估（表4.2）
    print("Running Fixed Angle Evaluation (Table 4.2)...")
    fixed_results_25db = run_fixed_angle_evaluation(model, snr_db=25)
    fixed_results_35db = run_fixed_angle_evaluation(model, snr_db=35)

    # 打印性能表格
    print_performance_table(fixed_results_25db, fixed_results_35db)

    # 2. 运行变化角度评估（图4.8和4.9）
    print("\nRunning Varying Angle Evaluation (Figures 4.8 and 4.9)...")

    # 25dB结果
    results_25db = {}
    for n_sources in [1, 2, 3]:
        true, pred, errors = run_varying_angle_evaluation(model, n_sources, snr_db=25)
        results_25db[n_sources] = (true, pred, errors)

    # 35dB结果
    results_35db = {}
    for n_sources in [1, 2, 3]:
        true, pred, errors = run_varying_angle_evaluation(model, n_sources, snr_db=35)
        results_35db[n_sources] = (true, pred, errors)

    # 3. 绘制所有结果
    plot_varying_angle_results(results_25db, results_35db)
    plot_error_analysis(results_25db, results_35db)

    print("\nAll evaluations completed! Check the 'plots' directory for generated figures.")


if __name__ == "__main__":
    main()