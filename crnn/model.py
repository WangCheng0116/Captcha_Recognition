import torch
from torch import nn
from torch.nn import functional as F

# class CNN(nn.Module):
#     def __init__(self, num_classes):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
#         self.max_pool1 = nn.MaxPool2d(2)
#         self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.max_pool2 = nn.MaxPool2d(2)
#         self.linear = nn.Linear(5120, 64)
#         self.dropout = nn.Dropout(0.2)
#         self.gru = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25)
#         self.out = nn.Linear(64, num_classes + 1)

#     def forward(self, images, labels=None, labels_mask=None):
#         bs, c, h, w = images.size()
#         # print(bs, c, h, w)
#         x = F.relu(self.conv1(images))
        # print(x.size())
#         x = self.max_pool1(x)
        # print(x.size())
#         x = F.relu(self.conv2(x))
        # print(x.size())
#         x = self.max_pool2(x) # 1, 64, 20, 195
        # print(x.size())
#         x = x.permute(0, 3, 1, 2) # 1, 195, 64, 20
        # print(x.size())
#         x = x.view(bs, x.size(1), -1) # 1, 195, 1280
        # print(x.size())
#         x = self.linear(x)
#         x = self.dropout(x) # 1, 195, 64
        # print(x.size())
#         x, _ = self.gru(x)
        # print(x.size())
#         x = self.out(x)
        # print(x.size())
#         x = x.permute(1, 0, 2)
        # print(x.size())
#         if labels is not None:
#             log_softmax_values = F.log_softmax(x, 2)
#             input_lengths = torch.full(
#                 size=(bs,), fill_value=log_softmax_values.size(0), dtype=torch.int32
#             )
#             # print(input_lengths)
#             if labels_mask is not None:
#                 target_lengths = labels_mask.sum(dim=1, dtype=torch.int32)
#             else:
#                 target_lengths = torch.full(
#                     size=(bs,), fill_value=labels.size(1), dtype=torch.int32
#                 )
#             # print(target_lengths)
#             loss = nn.CTCLoss(blank=0)(
#                 log_softmax_values, labels, input_lengths, target_lengths
#             )
#             return x, loss
#         return x, 


# class CNNLSTM(nn.Module):
#     def __init__(self, image_width, image_height, num_classes):
#         super(CNNLSTM, self).__init__()
#         # Rescaling (assuming input images are [0, 255] range, so divide by 255)
#         self.rescaling = nn.Identity()  # Placeholder, normalize in dataloader if needed

#         # Convolutional layers with batch normalization and max pooling
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.pool1 = nn.MaxPool2d((2, 2))

#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.pool2 = nn.MaxPool2d((2, 2))

#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.pool3 = nn.MaxPool2d((2, 1))  # Pooling only in the height dimension

#         # Fully connected layer
#         self.fc = nn.Linear((image_height // 8) * 256, 128)
#         self.dropout = nn.Dropout(0.4)

#         # Bidirectional LSTM layer
#         self.lstm = nn.LSTM(128, 128, bidirectional=True, num_layers=1, batch_first=True, dropout=0.25)

#         # Output layer
#         self.output = nn.Linear(256, num_classes + 1)  # Adding 1 for the CTC "blank" token

#     def forward(self, images, labels=None, labels_mask=None):
#         bs, c, h, w = images.size()

#         # Convolutional layers
#         x = F.relu(self.bn1(self.conv1(images)))
#         x = self.pool1(x)

#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.pool2(x)

#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.pool3(x)  # Shape: (bs, 256, h/4, w/8)

#         # Reshape for LSTM input
#         x = x.permute(0, 3, 1, 2)  # (bs, w/8, 256, h/4)
#         x = x.view(bs, x.size(1), -1)  # Shape: (bs, w/8, 256 * (h/4))

#         # Fully connected and dropout
#         x = self.fc(x)
#         x = self.dropout(x)

#         # Bidirectional LSTM
#         x, _ = self.lstm(x)  # Output shape: (bs, w/8, 256)

#         # Output layer for character probabilities
#         x = self.output(x)  # Shape: (bs, w/8, num_classes + 1)
#         x = x.permute(1, 0, 2)  # Shape for CTC Loss: (w/8, bs, num_classes + 1)

#         # Compute CTC Loss if labels are provided
#         if labels is not None:
#             log_softmax_values = F.log_softmax(x, 2)
#             input_lengths = torch.full((bs,), log_softmax_values.size(0), dtype=torch.int32)

#             if labels_mask is not None:
#                 target_lengths = labels_mask.sum(dim=1, dtype=torch.int32)
#             else:
#                 target_lengths = torch.full((bs,), labels.size(1), dtype=torch.int32)

#             loss = nn.CTCLoss(blank=0)(
#                 log_softmax_values, labels, input_lengths, target_lengths
#             )
#             return x, loss

#         return x

class CNNLSTM(nn.Module):
    def __init__(self, image_width, image_height, num_classes):
        super(CNNLSTM, self).__init__()
        # Rescaling (assuming input images are [0, 255] range, so divide by 255)
        self.rescaling = nn.Identity()  # Placeholder, normalize in dataloader if needed

        # Convolutional layers with batch normalization and max pooling
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d((2, 1))  # Pooling only in the height dimension

        # Fully connected layer
        self.fc = nn.Linear((image_height // 8) * 256, 128)
        self.dropout = nn.Dropout(0.4)

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(128, 128, bidirectional=True, num_layers=1, batch_first=True, dropout=0.25)

        # Output layer
        self.output = nn.Linear(256, num_classes + 1)  # Adding 1 for the CTC "blank" token

    def forward(self, images, labels=None, labels_mask=None):
        bs, c, h, w = images.size()
        # print(bs, c, h, w)

        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(images)))
        x = self.pool1(x)
        # print(x.size())

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        # print(x.size())

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)  # Shape: (bs, 256, h/4, w/8)
        # print(x.size())

        # Reshape for LSTM input
        x = x.permute(0, 3, 1, 2)  # (bs, w/8, 256, h/4)
        x = x.view(bs, x.size(1), -1)  # Shape: (bs, w/8, 256 * (h/4))
        # print(x.size())

        # Fully connected and dropout
        x = self.fc(x)
        x = self.dropout(x)
        # print(x.size())

        # Bidirectional LSTM
        x, _ = self.lstm(x)  # Output shape: (bs, w/8, 256)
        # print(x.size())
        # Output layer for character probabilities
        x = self.output(x)  # Shape: (bs, w/8, num_classes + 1)
        x = x.permute(1, 0, 2)  # Shape for CTC Loss: (w/8, bs, num_classes + 1)
        # print(x.size())
        # Compute CTC Loss if labels are provided
        if labels is not None:
            log_softmax_values = F.log_softmax(x, 2)
            input_lengths = torch.full((bs,), log_softmax_values.size(0), dtype=torch.int32)

            if labels_mask is not None:
                target_lengths = labels_mask.sum(dim=1, dtype=torch.int32)
            else:
                target_lengths = torch.full((bs,), labels.size(1), dtype=torch.int32)

            loss = nn.CTCLoss(blank=0)(
                log_softmax_values, labels, input_lengths, target_lengths
            )
            return x, loss

        return x

# cm = CNN(36)
# img = torch.randn(5, 1, 80, 480)
# label = torch.randint(1, 37, (5, 5))
# label[0][4] = -1
# label[0][3] = -1
# label_mask = label != -1
# x, loss = cm(img, label, label_mask)
# print(label_mask)