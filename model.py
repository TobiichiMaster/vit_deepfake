from modelscope.models import ViTForImageClassification
from modelscope.preprocessors import ViTImageProcessor

def get_model(num_classes=2, model_name='google/vit-base-patch16-224'):
    model = ViTForImageClassification.from_pretrained(model_name, num_classes=num_classes)
    processor = ViTImageProcessor.from_pretrained(model_name)
    return model, processor