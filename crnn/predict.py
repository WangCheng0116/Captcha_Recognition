import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import CNNLSTM  # Ensure CNNLSTM model architecture is defined
from sklearn import preprocessing
import config 
import albumentations
from pathlib import Path

# Load the label encoder
lbl_enc = preprocessing.LabelEncoder()
lbl_enc.classes_ = np.load("label_classes.npy", allow_pickle=True)  # Assuming you saved label classes during training

# Load the model and its weights
model = CNNLSTM(image_width=config.IMAGE_WIDTH, image_height=config.IMAGE_HEIGHT, num_classes=len(lbl_enc.classes_))
model.load_state_dict(torch.load("best_model.pth", weights_only=True))
model.to(config.DEVICE)
model.eval()  # Set to evaluation mode

# Function to preprocess the image
def preprocess_image(image):
    """
    This is for model without noise preprocessing
    """
    # augmentations = albumentations.Compose(
    #     [albumentations.Normalize(always_apply=True)])
    # # image = Image.open(image_path)
    # image = image.convert("L")

    # image = image.resize(
    #         (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), resample=Image.BILINEAR
    #     )

    # image = np.array(image).astype(np.float32)
    # image = np.expand_dims(image, axis=0)

    # augmented = augmentations(image=image)
    # image = augmented["image"]

    # return torch.tensor(image, dtype=torch.float)

    """
    This is for model with noise preprocessing
    """
    augmentations = albumentations.Compose(
            [albumentations.Normalize(always_apply=True)])
    image = image.convert("L")

    image = np.array(image).astype(np.float32)

    image[image == 0] = 255
    image[image < 255] = 0

    # Convert back to PIL Image for resizing
    image = Image.fromarray(image.astype(np.uint8))
    image = image.resize(
        (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), resample=Image.BILINEAR
    )
    image = np.array(image).astype(np.float32)

    image = np.expand_dims(image, axis=0)

    augmented = augmentations(image=image)
    image = augmented["image"]

    return torch.tensor(image, dtype=torch.float)


# Function to decode predictions
def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)  # Reshape predictions as needed
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, axis=2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k - 1
            if k == -1:
                temp.append("-")
            else:
                temp.append(encoder.inverse_transform([k])[0])
        tp = "".join(temp)
        cap_preds.append(tp)
    return cap_preds

def postprocess(pred_text):
    # Remove placeholder characters
    # Remove consecutive duplicate characters
    result = []
    for i, char in enumerate(pred_text):
        if char != "-":
            if i == 0 or char != pred_text[i - 1]:  # Only append if it's not a consecutive duplicate
                result.append(char)
    return ''.join(result)

# Prediction function
def predict(image):
    # Preprocess the image
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(config.DEVICE)
    
    with torch.no_grad():  # Disable gradient calculation
        output = model(image_tensor.unsqueeze(0))  # Add batch dimension
        predictions = decode_predictions(output, lbl_enc)
        
    return predictions[0]  # Return decoded prediction for the image


def evaluate(test_dir):
    test_path = Path(test_dir)
    correct_captchas = 0
    correct_chars = 0
    total_captchas = 0
    total_chars = 0

    for img_path in test_path.glob("*.png"):
        total_captchas += 1
        true_label = img_path.stem.split('-')[0].lower()
        total_chars += len(true_label)

        # Load and predict
        image = Image.open(str(img_path))
        predicted = postprocess(predict(image))

        # Calculate accuracy
        if predicted == true_label:
            correct_captchas += 1

        # Character-level accuracy
        for pred_char, true_char in zip(predicted, true_label):
            if pred_char == true_char:
                correct_chars += 1

    captcha_accuracy = correct_captchas / total_captchas
    char_accuracy = correct_chars / total_chars

    return {
        'captcha_accuracy': captcha_accuracy,
        'character_accuracy': char_accuracy
    }

# Example usage
# image_path = "C:\\Users\\kevin\\Desktop\\src(1)\\data\\test\\1fh4thm-0.png"
# prediction = postprocess(predict(Image.open(image_path)))
# print("Predicted text:", prediction)
# print(predict(Image.open(image_path)))

# Evaluate
results = evaluate("C:\\Users\\kevin\\Desktop\\src(1)\\data\\test")
print(f"CAPTCHA Accuracy: {results['captcha_accuracy']:.2%}")
print(f"Character Accuracy: {results['character_accuracy']:.2%}")

import numpy as np
import matplotlib.pyplot as plt

# # Load the saved losses
# losses = np.load('losses.npy', allow_pickle=True).item()

# train_losses = losses['train_losses']
# valid_losses = losses['valid_losses']

# print("Train Losses:", train_losses)
# print("Validation Losses:", valid_losses)

# # Plotting the train and validation losses
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label="Train Loss")
# plt.plot(valid_losses, label="Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Train vs Validation Loss")
# plt.legend()
# plt.show()
