import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_scheduler
from tqdm.auto import tqdm
import numpy as np
import mlflow
import mlflow.pytorch


from src.model_evaluation import calculate_metics


def train_model(
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    num_epochs: int,
    learning_rate: float,
    model_name: str,
    output_dir: str = "./models",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    
    """
        Train a Hugging Face Transformer model for sequence classification.
        
        Args:
        train_dataloader (DataLoader): DataLoader for the training set.
        val_dataloader (DataLoader): DataLoader for the validation set.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        model_name (str): Name of the pre-trained model (e.g., 'distilbert-base-uncased').
        output_dir (str): Directory to save the best model.
        device (str): Device to run the training on ('cuda' or 'cpu').

    Returns:
        tuple: A tuple containing:
            - model (AutoModelForSequenceClassification): The trained model.
            - history (dict): A dictionary containing training and validation loss/metrics per epoch.
    """
    

    print(f"Loading model {model_name} for sequence classification ...")
    # Load a pre-trained mode for sequence classification
    # num_label = 2 for binary classification
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2)
    model.to(device) # move model to the device
    
    
    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Define the learning rate scheduler
    # this helps in fine-tuning by gradually decreasing the learning-rate
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name = "linear",
        optimizer = optimizer,
        num_warmup_steps = 0,
        num_training_steps = num_training_steps
        
    )
    
    # Initialize variable to track best validation performance for model validation 
    best_val_f1 = -1
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy':[], 'val_precision':[], 'val_recall':[], 'val_f1_score':[]}
    
    print(f"Starting training for {num_epochs} epochs on device: {device}")
    
    for epoch in range(num_epochs):
        # == Training Loop ==
        
        model.train() # set up the model for training mode
        total_train_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training")
        
        for batch in progress_bar:
            # Move batch data to the correct device
            batch = {k : v.to(device) for k, v in batch.items()}
            outputs = model(**batch) # forward pass: calculate logits and loss
            loss = outputs.loss # The Hugging face calculate the loss if the labels are provided
            total_train_loss += loss.item()
            
            loss.backward() 
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = total_train_loss / len(train_dataloader)
        history['train_loss'].append(avg_train_loss)
        print(f"Epoch {epoch+1} - training Loss : {avg_train_loss:4f}")
        
        
        
        # == Validation loop
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1} validation")
        
        with torch.no_grad(): 
            for batch in progress_bar:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                total_val_loss += loss.item()
                
                # get predicted class by taking argmax of logits
                predictions = torch.argmax(logits, dim=1).cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                
                all_preds.extend(predictions)
                all_labels.extend(labels)
                
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_metrics = calculate_metics(np.array(all_preds), np.array(all_labels))

        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1_score'].append(val_metrics['f1_score'])
        
        print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}, "
              f"Accuracy: {val_metrics['accuracy']:.4f}, "
              f"Precision: {val_metrics['precision']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}, "
              f"F1-score: {val_metrics['f1_score']:.4f}")
        
        
        ## Log validation metrics with mlflow
        mlflow.log_metric("val_loss", avg_val_loss, step = epoch)
        mlflow.log_metric("val_accuracy", val_metrics['accuracy'], step = epoch)
        mlflow.log_metric("val_precision", val_metrics['precision'], step = epoch)
        mlflow.log_metric("val_recall", val_metrics['recall'], step = epoch)
        mlflow.log_metric("val_f1_score", val_metrics['f1_score'], step = epoch)
       
       # Save the best model based on F1-score
        if val_metrics['f1_score'] > best_val_f1:
           print(f"F1-score improved from {best_val_f1:.4f} to {val_metrics['f1_score']:.4f}. Saving model ...")
           best_val_f1 = val_metrics['f1_score']
           # Save model's state dict
        #    torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
           # Hugging Face also provided a conveninent save-pretrained method
        #    model.save_pretrained(f"{output_dir}/best_hf_model")
        print("Best F1-score achieved. Model will be logged via MLflow at the end of training.")

           
    print("Training complete")
    return model, history, best_val_f1



def evaluate_model(
    model: AutoModelForSequenceClassification,
    test_dataloader: torch.utils.data.DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
        Evaluate the trained model on the test set
        
        Args:
            model, test_dataloader, device
        Returns:
            dict: evaluation metrics
    """
    
    print("Starting final model evaluation on the test set")
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    progress_bar = tqdm(test_dataloader, desc="Test Evaluation")
    
    
    with torch.no_grad():
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            
            all_preds.extend(predictions)
            all_labels.extend(labels)
    test_metrics = calculate_metics(np.array(all_preds), np.array(all_labels))
    
    
    print("Final Test Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric.replace('_', ' ').capitalize()}: {value:.4f}")
        
    return test_metrics