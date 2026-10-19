import torch.nn as nn
from torchvision import models


class CustomCNN(nn.Module):
    """Compact baseline trained from scratch."""
    def __init__(self, num_classes=4):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout2d(.10),
            )
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(.35), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(name, pretrained=True, num_classes=4):
    if name == "custom_cnn":
        return CustomCNN(num_classes), None
    if name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model, "ImageNet1K_V1" if pretrained else None
    if name == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights, image_size=224)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model, "ImageNet1K_V1" if pretrained else None
    raise ValueError(f"Unknown model: {name}")


def freeze_backbone(model, name):
    for p in model.parameters():
        p.requires_grad = False
    head = model.classifier if name == "efficientnet_v2_s" else model.heads if name == "vit_b_16" else None
    if head is not None:
        for p in head.parameters():
            p.requires_grad = True


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True
