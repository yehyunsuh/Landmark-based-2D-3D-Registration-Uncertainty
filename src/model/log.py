import wandb
import numpy as np


def initiate_wandb(args):
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
    )


def log_results(train_loss, val_loss, mean_dist, mean_dice, best_mean_error, best_val_loss):
    wandb.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "mean_dist": mean_dist,
        "mean_dice": mean_dice,
        "best_mean_error": best_mean_error,
        "best_val_loss": best_val_loss,
    })


def log_selection_results(total_loss, best_loss):
    all_losses = np.concatenate(list(total_loss.values()))  # Flatten all arrays
    total_mean_loss = np.mean(all_losses)
    case_mean_losses = {f"{case_id} Mean Loss": np.mean(loss_array) for case_id, loss_array in total_loss.items()}

    best_all_loses = np.concatenate(list(best_loss.values()))  # Flatten all arrays
    best_total_mean_loss = np.mean(best_all_loses)
    best_case_mean_losses = {f"{case_id} Best Mean Loss": np.mean(loss_array) for case_id, loss_array in best_loss.items()}

    wandb.log({
        "Mean Loss": total_mean_loss,
        **case_mean_losses,
        "Best Mean Loss": best_total_mean_loss,
        **best_case_mean_losses
    })