import os
import pandas as pd


def save_csv(args, all_results, suffix=""):
    # ============================================================
    # SAVE RESULTS TO CSV (same columns as before)
    # ============================================================
    results_df = pd.DataFrame(all_results)

    # Difference columns
    results_df["pred_diff"]  = results_df["pred_err_all"]  - results_df["pred_err_filtered"]
    results_df["rot_diff"]   = results_df["rot_err_all"]   - results_df["rot_err_filtered"]
    results_df["trans_diff"] = results_df["trans_err_all"] - results_df["trans_err_filtered"]

    # Reorder columns
    results_df = results_df[
        [
            "image",
            "pred_err_all",
            "pred_err_filtered",
            "pred_diff",
            "rot_err_all",
            "rot_err_filtered",
            "rot_diff",
            "trans_err_all",
            "trans_err_filtered",
            "trans_diff",
            "pred_better",
            "pose_better",
        ]
    ]

    
    csv_path = os.path.join(
        args.vis_dir,
        f"prediction_{args.dropout_rate}",
        "final_results",
        f"test_results_summary_{args.top_k_landmarks}_{suffix}.csv"
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print("[Saved results CSV] →", csv_path)
    print()

    return results_df