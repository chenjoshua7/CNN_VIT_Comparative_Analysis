from tqdm import tqdm
import torch
from vit_helpers import save_model

class VitTrainer:
    def __init__(self, model, optimizer, device = None) -> None:
        self.model = model
        self.optimizer = optimizer
        if device:
            self.device = device
            self.model.to(device)
        self.val_accuracy = 0
        self.best_val_accuracy  = 0
        self.count = 0
        self.history = {"train_accuracy": [], "train_loss": [], "val_accuracy": [], "val_loss": []}

    def _calculate_metrics(running_loss, correct, total, outputs, y):
        ## Calculating Loss
        loss = outputs.loss
        running_loss += loss.item()
        ## Calculating Accuracy
        output = outputs.logits.argmax(dim=1)
        correct += (output == y).sum().item()
        total += y.size(0)
        return running_loss, correct/total
        
    def train(self, training_set, validation_set, epochs: int):
        for epoch in range(epochs):
            self.model.train()
            running_loss, train_correct, train_total = 0.0, 0, 0
            progress_bar = tqdm(enumerate(training_set), total=len(training_set), desc=f'Epoch {epoch+1}/{epochs}')
            
            for step, (x, y) in progress_bar:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(x, labels=y) 
                loss = outputs.loss

                if loss is not None:
                    loss.backward()
                self.optimizer.step()
                
                running_loss, train_acc = self._calculate_metrics(running_loss = running_loss,correct = train_correct, total = train_total, outputs = outputs, y=y)

                progress_bar.set_postfix({'train_loss': f'{running_loss / (step + 1):.4f}', 'train_acc': f'{train_acc:.4f}'})

            self.model.eval()
            with torch.no_grad():
                valid_loss, valid_correct, valid_total = 0.0, 0, 0
                for x, y in validation_set:
                    x, y = x.to(self.device), y.to(self.device)
                    outputs = self.model(x, labels=y)
                    valid_output = outputs.logits.argmax(dim=1)
                    valid_loss += outputs.loss.item()
                    valid_correct += (valid_output == y).sum().item()
                    valid_total += y.size(0)

                val_accuracy = valid_correct / valid_total
                valid_loss /= len(validation_set)
                print(f'Validation Loss: {valid_loss:.4f} | Accuracy: {val_accuracy:.4f}')

                # Save the best model
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    save_model(path=os.path.join(file_path,f'{epoch+1}_val_acc{val_accuracy:.4f}.pth'), model=model, epoch=epoch, optimizer=optimizer)
                    print("Saved Best Model")
                    count = 0
                else:
                    if EARLY_STOPPING:
                        count += 1
                    if count == 5:
                        print("Early Stopping Implemented")
                        break

                progress_bar.set_description(f'Epoch {epoch+1}/{EPOCHS} - Val Accuracy: {val_accuracy:.4f}')
        
            self.history["train_accuracy"].append(train_correct / train_total)
            self.history["train_loss"].append(running_loss / (step + 1))
            self.history["val_accuracy"].append(val_accuracy)
            self.history["val_loss"].append(valid_loss)
            
            history_path = os.path.join(file_path, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(history, f)
            

        
save_model(path=f'checkpoints/ViT/{epoch+1}_val_acc{val_accuracy:.4f}.pth', model=model, epoch=epoch, optimizer=optimizer)