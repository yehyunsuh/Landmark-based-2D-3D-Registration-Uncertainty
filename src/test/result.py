import os
import pandas as pd


def save_csv(args, all_results, suffix=""):
    # ============================================================
    # SAVE RESULTS TO CSV (same columns as before)
    # ============================================================
    results_df = pd.DataFrame(all_results)

    # Difference columns
    results_df["pred_diff_filtered"]  = results_df["pred_err_all"]  - results_df["pred_err_filtered"]
    results_df["rot_diff_filtered"]   = results_df["rot_err_all"]   - results_df["rot_err_filtered"]
    results_df["trans_diff_filtered"] = results_df["trans_err_all"] - results_df["trans_err_filtered"]

    # results_df["pred_diff_weighted"]  = results_df["pred_err_all"]  - results_df["pred_err_weighted"]
    results_df["rot_diff_weighted_ver1"]   = results_df["rot_err_all"]   - results_df["rot_err_weighted_ver1"]
    results_df["trans_diff_weighted_ver1"] = results_df["trans_err_all"] - results_df["trans_err_weighted_ver1"]

    results_df["rot_diff_weighted_ver2"]   = results_df["rot_err_all"]   - results_df["rot_err_weighted_ver2"]
    results_df["trans_diff_weighted_ver2"] = results_df["trans_err_all"] - results_df["trans_err_weighted_ver2"]

    results_df["rot_diff_weighted_ver3"]   = results_df["rot_err_all"]   - results_df["rot_err_weighted_ver3"]
    results_df["trans_diff_weighted_ver3"] = results_df["trans_err_all"] - results_df["trans_err_weighted_ver3"]

    results_df["pred_diff_gt"] = results_df["pred_err_all"] - results_df["pred_err_gt"]
    results_df["rot_diff_gt"]  = results_df["rot_err_all"]  - results_df["rot_err_gt"]
    results_df["trans_diff_gt"] = results_df["trans_err_all"] - results_df["trans_err_gt"]

    # Reorder columns
    # results_df = results_df[
    #     [
    #         "image",
    #         "pred_err_all",
    #         "pred_err_filtered",
    #         "pred_diff",
    #         "rot_err_all",
    #         "rot_err_filtered",
    #         "rot_diff",
    #         "trans_err_all",
    #         "trans_err_filtered",
    #         "trans_diff",
    #         "pred_better",
    #         "pose_better",
    #     ]
    # ]
            # all_results.append({
            #     "image": image_name,
            #     "pred_err_all": pred_err_all,
            #     "pred_err_filtered": pred_err_filtered,
            #     "pred_better": pred_better,

            #     "rot_err_all": rot_err_all,
            #     "rot_err_filtered": rot_err_filt,
            #     "rot_err_weighted": rot_err_w,

            #     "trans_err_all": trans_err_all,
            #     "trans_err_filtered": trans_err_filt,
            #     "trans_err_weighted": trans_err_w,
                
            #     "pose_better": pose_better,
            #     "pose_better_weighted": pose_better_w,
            # })

    results_df = results_df[
        [
            "image",
            "pred_err_all",
            "pred_err_filtered",
            "pred_diff_filtered",
            "pred_err_gt",
            "pred_diff_gt",

            "rot_err_all",
            "rot_err_filtered",
            "rot_diff_filtered",
            "rot_err_weighted_ver1",
            "rot_diff_weighted_ver1",
            "rot_err_weighted_ver2",
            "rot_diff_weighted_ver2",
            "rot_err_weighted_ver3",
            "rot_diff_weighted_ver3",
            "rot_err_gt",
            "rot_diff_gt",

            "trans_err_all",
            "trans_err_filtered",
            "trans_diff_filtered",
            "trans_err_weighted_ver1",
            "trans_diff_weighted_ver1",
            "trans_err_weighted_ver2",
            "trans_diff_weighted_ver2",
            "trans_err_weighted_ver3",
            "trans_diff_weighted_ver3",
            "trans_err_gt",
            "trans_diff_gt",
        ]
    ]

    
    csv_path = os.path.join(
        args.vis_dir,
        args.save_folder_name,
        "final_results",
        f"test_results_summary_{args.top_k_landmarks}_{suffix}.csv"
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print("[Saved results CSV] →", csv_path)
    print()

    return results_df