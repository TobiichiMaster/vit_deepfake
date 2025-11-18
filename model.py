import timm
def get_model(num_classes=2, arch='vit_base_patch16_224'):
    return timm.create_model(arch, pretrained=True, num_classes=num_classes)