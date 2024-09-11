import torch.optim as optim
import torch
from dataloader import load_data
from model import CNN
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import time
import pandas as pd

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy

def get_model_path(output_dir, run, training_time, epoch):
    model_name = f"{run}_{int(training_time)}_{epoch}.pth"
    return os.path.join(output_dir, model_name)

def get_plot_path(pic_output_dir, run, training_time, epoch):
    plot_name = f"{run}_{int(training_time)}_{epoch}.png"
    return os.path.join(pic_output_dir, plot_name)

def get_csv_path(csv_output_dir, run, training_time, epoch):
    csv_name = f"{run}_{int(training_time)}_{epoch}.csv"
    return os.path.join(csv_output_dir, csv_name)

if __name__ == '__main__':
    # parameters
    train_csv = "/path/to/your/train_features.csv"
    val_csv = "path/to/your/val_features.csv"
    batch_sizes = [16]
    learning_rates = [0.001]
    num_epochs = 1000
    min_epochs = 90  # Minimum epochs before starting early stopping
    patience = 20  # Early stopping patience
    output_dir = "/your/output/path"
    pic_output_dir = '/your/output/path/png'
    csv_output_dir = '/your/output/path/csv'

    # device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for run in range(1, 5):  # Repeat training 4 times
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                # load data
                train_loader, val_loader = load_data(train_csv, val_csv, batch_size=batch_size)

                model = CNN(class_dim=2).to(device)

                # loss function and optimiser
                criterion = nn.CrossEntropyLoss()
                # criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                train_losses = []
                train_accuracies = []
                val_losses = []
                val_accuracies = []

                best_val_loss = float('inf')
                best_val_accuracy = 0.0
                early_stopping_counter = 0

                start_time = time.time()
                # training
                for epoch in range(num_epochs):
                    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
                    val_loss, val_accuracy = validate(model, val_loader, criterion, device)

                    train_losses.append(train_loss)
                    train_accuracies.append(train_accuracy)
                    val_losses.append(val_loss)
                    val_accuracies.append(val_accuracy)

                    print(
                        f"Run: {run}, Batch Size: {batch_size}, Learning Rate: {learning_rate}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

                    # Check for early stopping
                    if epoch + 1 >= min_epochs:
                        if val_loss < best_val_loss or val_accuracy > best_val_accuracy:
                            best_val_loss = min(val_loss, best_val_loss)
                            best_val_accuracy = max(val_accuracy, best_val_accuracy)
                            early_stopping_counter = 0
                        else:
                            early_stopping_counter += 1
                            if early_stopping_counter >= patience:
                                print(f"Early stopping at epoch {epoch + 1}")
                                break

                end_time = time.time()
                training_time = end_time - start_time
                print(f'Training completed in {training_time:.2f} seconds')

                # save model
                os.makedirs(output_dir, exist_ok=True)
                model_save_path = get_model_path(output_dir, run, training_time, epoch + 1)
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved to {model_save_path}")

                # save plot
                os.makedirs(pic_output_dir, exist_ok=True)
                plot_save_path = get_plot_path(pic_output_dir, run, training_time, epoch + 1)

                epochs = range(1, len(train_losses) + 1)
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2, 1)
                plt.plot(epochs, train_losses, 'r-', label='Train Loss')
                plt.plot(epochs, val_losses, 'b-', label='Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title('Train and Validation Loss')
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.plot(epochs, train_accuracies, 'r-', label='Train Accuracy')
                plt.plot(epochs, val_accuracies, 'b-', label='Validation Accuracy')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.title('Train and Validation Accuracy')
                plt.legend()

                plt.tight_layout()
                plt.savefig(plot_save_path)
                plt.show()
                print(f"Plot saved to {plot_save_path}")

                # save csv
                os.makedirs(csv_output_dir, exist_ok=True)
                csv_save_path = get_csv_path(csv_output_dir, run, training_time, epoch + 1)
                df = pd.DataFrame({
                    'Epoch': epochs,
                    'Train Loss': train_losses,
                    'Train Accuracy': train_accuracies,
                    'Validation Loss': val_losses,
                    'Validation Accuracy': val_accuracies
                })
                df.to_csv(csv_save_path, index=False)
                print(f"CSV saved to {csv_save_path}")
