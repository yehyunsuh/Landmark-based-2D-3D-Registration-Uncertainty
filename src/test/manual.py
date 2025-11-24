import torch


def manual_translation(specimen_id, svd):
    if specimen_id == '17-1882':
        manual_translations_list = torch.tensor([[0.0, svd, -0.0]])
    elif specimen_id == '17-1905':
        manual_translations_list = torch.tensor([[0.0, svd, 0.0]])
    elif specimen_id == '18-0725':
        manual_translations_list = torch.tensor([[0.0, svd, 70.0]])
    elif specimen_id == '18-1109':
        manual_translations_list = torch.tensor([[0.0, svd, 50.0]])
    elif specimen_id == '18-2799':
        manual_translations_list = torch.tensor([[0.0, svd, 40.0]])
    elif specimen_id == '18-2800':
        manual_translations_list = torch.tensor([[0.0, svd, 0.0]])
    else:
        raise ValueError(f"Unknown specimen ID {specimen_id}")
    
    return manual_translations_list