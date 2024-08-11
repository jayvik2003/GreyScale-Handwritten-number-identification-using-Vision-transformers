from Cnn import CNN
from Vit_former import VisionTransformer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


vit_model = VisionTransformer(img_size=32, patch_size=4, num_classes=10, embed_dim=32, depth=2, num_heads=8, mlp_dim=64)

#CNN_model = CNN()

# Compute the number of parameters
num_parameters = count_parameters(vit_model)
#num_parameters = count_parameters(CNN_model)

print(f"Number of parameters: {num_parameters}")
