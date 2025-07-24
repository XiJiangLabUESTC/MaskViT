import timm
from timm.models.vision_transformer import VisionTransformer
from timm.models.registry import register_model
from .MaskViT import MaskVisionTransformer

@register_model
def vit_tiny_patch16_84(num_classes=1000, **kwargs):
    model = VisionTransformer(
        img_size=84,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        num_classes=num_classes
    )
    return model

@register_model
def vit_tiny_patch16_84_in21k(num_classes=1000, **kwargs):
    model = VisionTransformer(
        img_size=84,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        num_classes=num_classes
    )
    dino_pretrained_model = timm.create_model('vit_tiny_patch16_224_in21k', pretrained=True, num_classes=num_classes)
    model_dict = dino_pretrained_model.state_dict()
    del_key = ['pos_embed']
    for key in del_key:
        del model_dict[key]
    model.load_state_dict(model_dict, strict=False)
    return model

@register_model
def maskvit_tiny_patch16_84_in21k(num_classes=1000, mask_rate=0.5, mask_layer=10,**kwargs):
    model = MaskVisionTransformer(
        img_size=84,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        num_classes=num_classes,
        mask_rate=mask_rate,
        mask_layer=mask_layer
    )
    dino_pretrained_model = timm.create_model('vit_tiny_patch16_224_in21k', pretrained=True, num_classes=num_classes)
    model_dict = dino_pretrained_model.state_dict()
    del_key = ['pos_embed']
    for key in del_key:
        del model_dict[key]
    model.load_state_dict(model_dict, strict=False)
    return model

@register_model
def maskvit_tiny_patch16_84(num_classes=1000, mask_rate=0.5, mask_layer=10,**kwargs):
    model = MaskVisionTransformer(
        img_size=84,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        num_classes=num_classes,
        mask_rate=mask_rate,
        mask_layer=mask_layer
    )
    return model