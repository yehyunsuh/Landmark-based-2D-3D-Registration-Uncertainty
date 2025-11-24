import torch.nn as nn
import segmentation_models_pytorch as smp


def UNet_with_dropout(args, device):
    model = smp.Unet(
        encoder_name='resnet101',
        encoder_depth=args.encoder_depth,
        encoder_weights='imagenet',
        decoder_channels=args.decoder_channels,
        in_channels=3,
        classes=args.n_landmarks,
        activation=None
    )
    dropout_p = args.dropout_rate
    for block in model.decoder.blocks:
        block.dropout = nn.Dropout2d(dropout_p)
        orig_forward = block.forward
        def new_forward(*a, orig_forward=orig_forward, block=block, **kw):
            x = orig_forward(*a, **kw)
            return block.dropout(x)
        block.forward = new_forward
    return model.to(device)