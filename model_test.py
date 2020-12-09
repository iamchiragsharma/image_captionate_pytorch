import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms as T
from PIL import Image

inception = models.inception_v3(pretrained=True, aux_logits=True)

# model = nn.Sequential(
#     *list(inception.children())[:-1]
# )

inception.fc = nn.Identity()
inception.eval()

transform = T.Compose([T.Resize(334),T.CenterCrop(299),T.ToTensor()])

test_img = transform(Image.open("flickr8k/images/667626_18933d713e.jpg").convert("RGB")).unsqueeze(0)
print(test_img.shape)
outputs = inception(test_img)

print(inception.fc.input_feat)

print(outputs.shape)
