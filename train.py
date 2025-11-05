import os 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from data_loader import ImSituDataset
from models import SituationRecognizer

# Hyperparameters and Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = './data/imsitu'
SAVE_DIR = './saved_models'
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10


# Create the save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)


def calculate_metrics(verb_logits, role_logits, verb_labels, role_labels):
    """Calculates verb accuracy and role-noun P/R/F1 score for a batch."""
    
    # Verb Accuracy
    verb_preds = torch.argmax(verb_logits, dim=1)
    verb_correct = (verb_preds == verb_labels).sum().item()
    
    # --- Role-Noun P/R/F1 ---
    role_preds = torch.argmax(role_logits, dim=2) # Shape: (batch_size, num_roles)
    
    true_positives = 0
    predicted_positives = 0
    actual_positives = 0
    
    for i in range(role_labels.shape[0]): # Iterate over each item in the batch
        # We only evaluate on roles that are relevant for the ground truth verb
        relevant_roles_mask = role_labels[i] != -1
        
        # Ground truth (role_idx, noun_idx) pairs
        gt_roles = torch.where(relevant_roles_mask)[0]
        gt_nouns = role_labels[i][relevant_roles_mask]
        gt_pairs = set(zip(gt_roles.tolist(), gt_nouns.tolist()))
        
        # Predicted (role_idx, noun_idx) pairs for the relevant roles
        pred_nouns_for_relevant_roles = role_preds[i][relevant_roles_mask]
        pred_pairs = set(zip(gt_roles.tolist(), pred_nouns_for_relevant_roles.tolist()))
        
        true_positives += len(gt_pairs.intersection(pred_pairs))
        predicted_positives += len(pred_pairs)
        actual_positives += len(gt_pairs)
        
    return verb_correct, true_positives, predicted_positives, actual_positives


def train(model, dataloader, optimizer, verb_criterion, role_criterion, device):
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training", unit="batch")

    for batch in progress_bar:
        images = batch["image"].to(device)
        verb_labels = batch["verb_idx"].to(device)
        role_labels = batch["noun_indices"].to(device) #this is shaped (batch_size, num_roles)

        #forward pass
        optimizer.zero_grad()
        predictions = model(images)


        verb_loss = verb_criterion(predictions['verb_logits'], verb_labels)

        #we need to do slight reshaping for role loss calculations
        #pred is (batch, num_roles, num_nouns), we need to convert it to -> (batch*num_roles, num_nouns)
        # labels is in shape (batch, num_roles) -> (batch*num_roles,)
        num_nouns = predictions['role_logits'].shape[-1]
        role_loss = role_criterion(
            predictions['role_logits'].view(-1, num_nouns),
            role_labels.view(-1)
        )


        loss = verb_loss + role_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item)

    
    return total_loss/len(dataloader)








@torch.no_grad()
def evaluate(model, dataloader, verb_criterion, role_criterion, device):
    model.eval()
    total_loss = 0

    total_verb_correct = 0
    total_tp, total_pp, total_ap = 0, 0, 0 # true positives, predicted positives, actual positives

    progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")

    for batch in progress_bar:
        images = batch["image"].to(device)
        verb_labels = batch["verb_idx"].to(device)
        role_labels = batch["noun_indices"].to(device)
        
        #forward pass
        predictions = model(images)

        # Calculate loss
        verb_loss = verb_criterion(predictions['verb_logits'], verb_labels)
        num_nouns = predictions['role_logits'].shape[-1]
        role_loss = role_criterion(predictions['role_logits'].view(-1, num_nouns), role_labels.view(-1))
        loss = verb_loss + role_loss
        total_loss += loss.item()


        # Calculate metrics
        verb_correct, tp, pp, ap = calculate_metrics(
            predictions['verb_logits'], predictions['role_logits'], verb_labels, role_labels
        )
        total_verb_correct += verb_correct
        total_tp += tp
        total_pp += pp
        total_ap += ap

    # --- Aggregate and calculate final metrics ---
    avg_loss = total_loss / len(dataloader)
    verb_accuracy = (total_verb_correct / len(dataloader.dataset)) * 100
    
    # Role-Noun Metrics
    precision = total_tp / total_pp if total_pp > 0 else 0
    recall = total_tp / total_ap if total_ap > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return avg_loss, verb_accuracy, precision, recall, f1_score


def main():
    print(f"Using device: {DEVICE}")

    print("Loading datasets...")
    train_dataset = ImSituDataset(data_dir=DATA_DIR, split='train')
    val_dataset = ImSituDataset(data_dir=DATA_DIR, split='dev')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print("Datasets loaded.")


    print("Initializing model...")
    model = SituationRecognizer(
        num_verbs=len(train_dataset.verb_to_idx),
        num_roles=train_dataset.num_roles,
        num_nouns=train_dataset.num_nouns
    ).to(DEVICE)


    # Verb loss is a standard classification loss
    verb_criterion = nn.CrossEntropyLoss()

    # Role loss ignores the '-1' labels for non-applicable roles
    role_criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # We only train the new prediction heads, not the frozen backbone
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("Model initialized.")




    # Training Loop
    best_f1 = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

        train_loss = train(model, train_loader, optimizer, verb_criterion, role_criterion, DEVICE)
        
        val_loss, verb_acc, precision, recall, f1 = evaluate(model, val_loader, verb_criterion, role_criterion, DEVICE)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Verb Acc:   {verb_acc:.2f}%")
        print(f"  Role P:     {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")


        if f1 > best_f1:
            best_f1 = f1
            save_path = os.path.join(SAVE_DIR, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved to {save_path} (F1: {best_f1:.4f})")
            
    print("\n--- Training Complete ---")
    print(f"Best validation F1 score: {best_f1:.4f}")


if __name__ == '__main__':
    main()
