import os
import seaborn as sns
import matplotlib.pyplot as plt


def plot_overall_trends(args, results_df, plot_dir, suffix=""):
    # ============================================================
    # REUSE PLOTTING LOGIC (updated: 3 columns = pred/rot/trans)
    # ============================================================
    df = results_df.copy()

    # Prediction columns for plotting
    df["pred_all"]  = df["pred_err_all"]
    df["pred_filt"] = df["pred_err_filtered"]

    # Rotation / translation columns for plotting
    df["rot_all"]   = df["rot_err_all"]
    df["rot_filt"]  = df["rot_err_filtered"]
    df["trans_all"] = df["trans_err_all"]
    df["trans_filt"] = df["trans_err_filtered"]

    # Differences
    df["pred_diff"]  = df["pred_filt"]  - df["pred_all"]
    df["rot_diff"]   = df["rot_filt"]   - df["rot_all"]
    df["trans_diff"] = df["trans_filt"] - df["trans_all"]

    sns.set(style="whitegrid", font_scale=1.4)

    # ------------------------------------------------
    # 1) PAIRED DIFFERENCE SCATTER PLOTS
    #    cols: prediction_diff, rotation_diff, translation_diff
    # ------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(24, 6))

    # --- Prediction diff ---
    axs[0].axhline(0, color="black", linewidth=1)
    sns.scatterplot(data=df, x=df.index, y="pred_diff", s=60, ax=axs[0])
    axs[0].set_title("Prediction Dist Error: Filtered − All", fontsize=16)
    axs[0].set_ylabel("Δ Projection Error (px)", fontsize=14)
    axs[0].set_xlabel("Image Index", fontsize=14)

    text_pred = (
        "How to interpret:\n"
        "• Δ < 0 → filtering improved 2D prediction\n"
        "• Δ > 0 → filtering worsened prediction\n"
        f"• % Improved: {(df['pred_diff'] < 0).mean()*100:.1f}%\n"
        f"• Mean Δ: {df['pred_diff'].mean():.3f} px"
    )
    axs[0].text(
        0.02, 0.98, text_pred,
        transform=axs[0].transAxes,
        fontsize=12,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9)
    )

    # --- Rotation diff ---
    axs[1].axhline(0, color="black", linewidth=1)
    sns.scatterplot(data=df, x=df.index, y="rot_diff", s=60, ax=axs[1])
    axs[1].set_title("Rotation Error: Filtered − All", fontsize=16)
    axs[1].set_ylabel("Δ Rotation Error (deg L2)", fontsize=14)
    axs[1].set_xlabel("Image Index", fontsize=14)

    text_rot = (
        "How to interpret:\n"
        "• Δ < 0 → filtering improved rotation\n"
        "• Δ > 0 → filtering worsened rotation\n"
        f"• % Improved: {(df['rot_diff'] < 0).mean()*100:.1f}%\n"
        f"• Mean Δ: {df['rot_diff'].mean():.3f}"
    )
    axs[1].text(
        0.02, 0.98, text_rot,
        transform=axs[1].transAxes,
        fontsize=12,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9)
    )

    # --- Translation diff ---
    axs[2].axhline(0, color="black", linewidth=1)
    sns.scatterplot(data=df, x=df.index, y="trans_diff", s=60, ax=axs[2])
    axs[2].set_title("Translation Error: Filtered − All", fontsize=16)
    axs[2].set_ylabel("Δ Translation Error (mm L2)", fontsize=14)
    axs[2].set_xlabel("Image Index", fontsize=14)

    text_trans = (
        "How to interpret:\n"
        "• Δ < 0 → filtering improved translation\n"
        "• Δ > 0 → filtering worsened translation\n"
        f"• % Improved: {(df['trans_diff'] < 0).mean()*100:.1f}%\n"
        f"• Mean Δ: {df['trans_diff'].mean():.3f}"
    )
    axs[2].text(
        0.02, 0.98, text_trans,
        transform=axs[2].transAxes,
        fontsize=12,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9)
    )

    plt.tight_layout()
    plt.savefig(os.path.join(
        plot_dir,
        f"overall_difference_trends_{args.top_k_landmarks}_{suffix}.png"
    ))
    plt.close()

    # ------------------------------------------------
    # 2) BOX PLOTS
    #    cols: prediction, rotation, translation (All vs Filtered)
    # ------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(24, 6))

    # Prediction
    sns.boxplot(data=df[["pred_all", "pred_filt"]], ax=axs[0])
    axs[0].set_title("Prediction Distance Error (All vs Filtered)", fontsize=16)
    axs[0].set_xticks([0, 1])
    axs[0].set_xticklabels(["All LMs", "Filtered"], fontsize=13)

    text_pred_box = (
        "Interpretation:\n"
        "• Lower median = more accurate 2D prediction\n"
        f"• Median(All):   {df['pred_all'].median():.2f} px\n"
        f"• Median(Filt): {df['pred_filt'].median():.2f} px"
    )
    axs[0].text(
        0.98, 0.98, text_pred_box, ha="right", va="top",
        transform=axs[0].transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9)
    )

    # Rotation
    sns.boxplot(data=df[["rot_all", "rot_filt"]], ax=axs[1])
    axs[1].set_title("Rotation Error (All vs Filtered)", fontsize=16)
    axs[1].set_xticks([0, 1])
    axs[1].set_xticklabels(["All LMs", "Filtered"], fontsize=13)

    text_rot_box = (
        "Interpretation:\n"
        "• Lower median = better rotation accuracy\n"
        "• Smaller box = more stable rotation\n"
        f"• Median(All):   {df['rot_all'].median():.3f}\n"
        f"• Median(Filt): {df['rot_filt'].median():.3f}"
    )
    axs[1].text(
        0.98, 0.98, text_rot_box, ha="right", va="top",
        transform=axs[1].transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9)
    )

    # Translation
    sns.boxplot(data=df[["trans_all", "trans_filt"]], ax=axs[2])
    axs[2].set_title("Translation Error (All vs Filtered)", fontsize=16)
    axs[2].set_xticks([0, 1])
    axs[2].set_xticklabels(["All LMs", "Filtered"], fontsize=13)

    text_trans_box = (
        "Interpretation:\n"
        "• Lower median = better translation accuracy\n"
        "• Smaller box = more stable translation\n"
        f"• Median(All):   {df['trans_all'].median():.3f}\n"
        f"• Median(Filt): {df['trans_filt'].median():.3f}"
    )
    axs[2].text(
        0.98, 0.98, text_trans_box, ha="right", va="top",
        transform=axs[2].transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9)
    )

    plt.tight_layout()
    plt.savefig(os.path.join(
        plot_dir,
        f"overall_boxplots_{args.top_k_landmarks}_{suffix}.png"
    ))
    plt.close()

    # ------------------------------------------------
    # 3) HISTOGRAMS
    #    cols: prediction_diff, rotation_diff, translation_diff
    # ------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(24, 6))

    # Prediction diff histogram
    sns.histplot(df["pred_diff"], kde=True, ax=axs[0])
    axs[0].axvline(0, color="black")
    axs[0].set_title("Δ Prediction Error Histogram", fontsize=16)

    text_pred_hist = (
        "Meaning of distribution:\n"
        "• Center < 0 → filtered prediction is better overall\n"
        "• Spread indicates stability of filtering\n"
        f"• Mean Δ: {df['pred_diff'].mean():.3f}\n"
        f"• Median Δ: {df['pred_diff'].median():.3f}"
    )
    axs[0].text(
        0.98, 0.98, text_pred_hist,
        ha="right", va="top",
        transform=axs[0].transAxes,
        fontsize=12,
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.9)
    )

    # Rotation diff histogram
    sns.histplot(df["rot_diff"], kde=True, ax=axs[1])
    axs[1].axvline(0, color="black")
    axs[1].set_title("Δ Rotation Error Histogram", fontsize=16)

    text_rot_hist = (
        "Meaning of distribution:\n"
        "• Center < 0 → filtering improves rotation on average\n"
        "• Long right tail → rotation hurts in some cases\n"
        f"• Mean Δ: {df['rot_diff'].mean():.3f}\n"
        f"• Median Δ: {df['rot_diff'].median():.3f}"
    )
    axs[1].text(
        0.98, 0.98, text_rot_hist,
        ha="right", va="top",
        transform=axs[1].transAxes,
        fontsize=12,
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.9)
    )

    # Translation diff histogram
    sns.histplot(df["trans_diff"], kde=True, ax=axs[2])
    axs[2].axvline(0, color="black")
    axs[2].set_title("Δ Translation Error Histogram", fontsize=16)

    text_trans_hist = (
        "Meaning of distribution:\n"
        "• Center < 0 → filtering improves translation on average\n"
        "• Long right tail → translation hurts in some cases\n"
        f"• Mean Δ: {df['trans_diff'].mean():.3f}\n"
        f"• Median Δ: {df['trans_diff'].median():.3f}"
    )
    axs[2].text(
        0.98, 0.98, text_trans_hist,
        ha="right", va="top",
        transform=axs[2].transAxes,
        fontsize=12,
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.9)
    )

    plt.tight_layout()
    plt.savefig(os.path.join(
        plot_dir,
        f"overall_histograms_{args.top_k_landmarks}_{suffix}.png"
    ))
    plt.close()