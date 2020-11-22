import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from get_loader import get_loader

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, trainCNN = False):
        super(EncoderCNN, self).__init__()

        self.trainCNN = trainCNN
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc == nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    
    def forward(self, images):
        features = self.inception(images)

        for name,param  in self.inception.parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.require_grad = True
            else:
                param.require_grad = self.trainCNN

        return self.dropout(self.relu(features))

transform = transforms.Compose([
    transforms.Resize((356, 356)),
    transforms.RandomCrop((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))
])

train_loader, dataset = get_loader(
    root_folder = 'flickr8k/images',
    annotation_file = 'flickr8k/captions.txt',
    transform=transform,
    num_workers = 2
)

encoderCNN = EncoderCNN(256)

for epoch in range(10):
    for idx, (images, captions) in enumerate(train_loader):
        outputs = encoderCNN(images)
        break
    break