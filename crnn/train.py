import os
import glob
import numpy as np
import torch

from sklearn import preprocessing
import config 
import dataset
import engine
from model import CNNLSTM
from pprint import pprint

def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)
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

def run_training():
    # Get the image files, excluding duplicate images
    train_image_files = [f for f in glob.glob(os.path.join(config.TRAIN_DATA_DIR, "*.png")) if "(1)" not in os.path.basename(f)]
    test_image_files = [f for f in glob.glob(os.path.join(config.TEST_DATA_DIR, "*.png")) if "(1)" not in os.path.basename(f)]

    # Prepare target labels
    train_targets_orig = [x.split("\\")[-1][:-6] for x in train_image_files]
    test_targets_orig = [x.split("\\")[-1][:-6] for x in test_image_files]

    train_targets = [[c for c in x] for x in train_targets_orig]
    test_targets = [[c for c in x] for x in test_targets_orig]

    train_targets_flat = [c for clist in train_targets for c in clist]
    test_targets_flat = [c for clist in test_targets for c in clist]

    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(train_targets_flat)

    # Save label classes for later 
    
    np.save("label_classes.npy", lbl_enc.classes_)

    # Encode and pad targets
    train_targets_enc = [lbl_enc.transform(x) + 1 for x in train_targets]
    test_targets_enc = [lbl_enc.transform(x) + 1 for x in test_targets]

    train_max_length = max(len(seq) for seq in train_targets_enc)
    train_targets_enc_padded = np.array([np.pad(seq, (0, train_max_length - len(seq)), constant_values=-1) for seq in train_targets_enc])
    test_targets_enc_padded = np.array([np.pad(seq, (0, train_max_length - len(seq)), constant_values=-1) for seq in test_targets_enc])

    # Datasets and DataLoaders
    train_dataset = dataset.ClassificationDataset(
        image_paths=train_image_files,
        targets=train_targets_enc_padded,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True, pin_memory=True, persistent_workers=True
    )
    
    test_dataset = dataset.ClassificationDataset(
        image_paths=test_image_files,
        targets=test_targets_enc_padded,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=False, pin_memory=True, persistent_workers=True
    )

    # Model, optimizer, and scheduler setup
    model = CNNLSTM(image_width=config.IMAGE_WIDTH, image_height=config.IMAGE_HEIGHT, num_classes=len(lbl_enc.classes_))
    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.3, verbose=True
    )

    # Early stopping and model saving setup
    best_valid_loss = float("inf")
    patience = 15
    wait = 0
    save_path = "best_model.pth"

    train_losses = []  # List to store training losses
    valid_losses = []  # List to store validation losses

    for epoch in range(config.EPOCHS):
        # Training phase
        train_loss = engine.train_fn(model, train_loader, optimizer)
        train_losses.append(train_loss)
        
        # Validation phase
        valid_preds, valid_loss = engine.eval_fn(model, test_loader)
        valid_losses.append(valid_loss)
        
        # Decode predictions for display
        valid_original = []
        for j in valid_preds:
            current_preds = decode_predictions(j, lbl_enc)
            valid_original.extend(current_preds)
        
        # Print sample predictions
        pprint(list(zip(test_targets_orig, valid_original))[:10])
        print(f"Epoch={epoch}, Train Loss={train_loss}, Valid Loss={valid_loss}")
        
        # Check for improvement
        if valid_loss < best_valid_loss:
            print(f"Validation loss improved from {best_valid_loss} to {valid_loss}. Saving model...")
            best_valid_loss = valid_loss
            wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1
            print(f"No improvement in validation loss. Early stopping wait counter: {wait}/{patience}")

        # Early stopping condition
        if wait >= patience:
            print("Early stopping triggered.")
            break

        # Update learning rate if needed
        scheduler.step(valid_loss)

    np.save('losses.npy', {'train_losses': train_losses, 'valid_losses': valid_losses})
    print("Losses saved as 'losses.npy'.")
    print("Training complete. Best model saved as 'best_model.pth' and label classes saved as 'label_classes.npy'.")

if __name__ == "__main__":
    run_training()
