from model.backbones.swin.swin_transformer import SwinTransformer
from model.backbones.vit_pytorch import TransReID
from functools import partial
import torch.nn as nn

'''
some parameters taken from
https://github.com/microsoft/Swin-Transformer/blob/main/configs/swin_base_patch4_window7_224.yaml

MODEL:
  TYPE: swin
  NAME: swin_base_patch4_window7_224
  DROP_PATH_RATE: 0.5
  SWIN:
    EMBED_DIM: 128
    DEPTHS: [ 2, 2, 18, 2 ]
    NUM_HEADS: [ 4, 8, 16, 32 ]
    WINDOW_SIZE: 7
'''

# TODO: replace TransReID constuctor ?
def swin_base_patch4_window7_224(img_size=(256, 128), stride_size=16, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, camera=0, view=0,local_feature=False,sie_xishu=1.5, **kwargs):
    model = TransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=128, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,\
        camera=camera, view=view, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  sie_xishu=sie_xishu, local_feature=local_feature, **kwargs)

    return model
