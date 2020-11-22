import torch
import torch.nn as nn
import torchvision.models as models

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


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0) #unsqueezing to add timesteps to features

        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)

        return outputs


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)


    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_len = 50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_len):
                hiddens, states = self.decoderRNN.lstm(x, states)
                outputs = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = outputs.argmax(1)

                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == '<EOS>':
                    break

            return [vocabulary.itos[idx] for idx in result_caption]

            


        


