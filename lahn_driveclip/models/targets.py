def get_last_visual_transformer_block(model):
    visual = model.clip.visual
    if not hasattr(visual, "transformer"):
        raise ValueError("Localization requires a ViT visual encoder.")
    return visual.transformer.resblocks[-1]
