import os

from src.data_processng import load_and_preprocess_data
from src.model_training import train_model, evaluate_model
import torch
import mlflow
import mlflow.pytorch
from datetime import datetime 
import pandas as pd



from src.inference import load_mlflow_model_and_tokenizer, predict_sentiment # NEW

# Define MLflow experiment name
MLFLOW_EXPERIMENT_NAME = "Sentiment Analysis IMDb"
MLFLOW_MODEL_NAME = "IMDbSentimentClassifier" # Name for the registered model
MLFLOW_TRACKING_URI = "file:./mlruns" # Local tracking



MLFLOW_EXPERIMENT_NAME = "Sentiment Analysis IMDb"
MLFLOW_MODEL_NAME = 'IMDbSentimentClassifier'


def main():
    """
    Main function to orchestrate the ML pipeline.
    """
    print("Starting ML Sentiment Analysis Project...")

    # --- Configuration ---
    MODEL_NAME = 'distilbert-base-uncased'
    MAX_LENGTH = 128
    NUM_EPOCHS = 3
    LEARNING_RATE = 2e-5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running on device: {DEVICE}")

    # --- MLflow Setup ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Start an MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # Log parameters
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("max_length", MAX_LENGTH)
        mlflow.log_param("num_epochs", NUM_EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("device", DEVICE)
        mlflow.log_param("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # --- Stage 1: Data Loading and Preprocessing ---
        print("\n--- Stage 1: Loading and Preprocessing Data ---")
        train_dataloader, val_dataloader, test_dataloader, tokenizer = load_and_preprocess_data(
            model_name=MODEL_NAME,
            max_length=MAX_LENGTH
        )
        print(f"Number of training batches: {len(train_dataloader)}")
        print(f"Number of validation batches: {len(val_dataloader)}")
        print(f"Number of test batches: {len(test_dataloader)}")


        # --- Stage 2: Model Training ---
        print("\n--- Stage 2: Training Model ---")
        trained_model, training_history, best_val_f1 = train_model(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            model_name=MODEL_NAME,
            device=DEVICE
        )

        mlflow.log_metric("best_val_f1_score", best_val_f1)
        print(f"Best Validation F1-score for this run: {best_val_f1:.4f}")

        # --- Stage 3: Model Evaluation (on Test Set) ---
        print("\n--- Stage 3: Evaluating Model on Test Set ---")
        final_test_metrics = evaluate_model(
            model=trained_model,
            test_dataloader=test_dataloader,
            device=DEVICE
        )

        print("\nProject pipeline execution finished.")
        print(f"Final Test Accuracy: {final_test_metrics['accuracy']:.4f}")

        # --- MLflow Model Logging and Registration ---
        # Log the trained PyTorch model as an MLflow artifact
        print(f"\nLogging model to MLflow and registering as '{MLFLOW_MODEL_NAME}'...")

        # We will log the tokenizer as a separate artifact to ensure it's easily retrievable
        # Save tokenizer to a temp directory within MLflow's run artifact directory
        temp_tokenizer_dir = os.path.join(mlflow.get_artifact_uri(), "tokenizer")
        tokenizer.save_pretrained(temp_tokenizer_dir)
        print(f"Tokenizer saved to artifact path: {temp_tokenizer_dir}")
        mlflow.log_artifacts(temp_tokenizer_dir, artifact_path="tokenizer") # Log the directory

        # Log the PyTorch model
        mlflow.pytorch.log_model(
            pytorch_model=trained_model,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME,
            # input_example={
            #     "input_ids": torch.tensor([[101, 2000, 2003, 102, 0, 0]]),
            #     "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0]])
            # },
            # Explicitly list requirements for robust deployment
            pip_requirements=[
                "torch",
                "transformers",
                "datasets", # needed for some HF functionality even in inference
                "scikit-learn",
                "pandas",
                "numpy"
            ]
        )
        print("Model logged and registered successfully.")

    # --- Stage 4: Inference Example (using the just-registered model) ---
    print("\n--- Stage 4: Demonstrating Inference ---")
    # For demonstration, we'll try to load the latest version of the registered model.
    # In a real scenario, you might specify a production-ready version.
    model_uri_for_inference = f"models:/{MLFLOW_MODEL_NAME}/latest"
    print(f"Attempting to load model for inference from: {model_uri_for_inference}")

    try:
        inference_model, inference_tokenizer, inference_device = load_mlflow_model_and_tokenizer(model_uri_for_inference)
        print(f"Inference model and tokenizer loaded. Device: {inference_device}")

        sample_review = "This is an absolutely amazing film with incredible performances!"
        prediction = predict_sentiment(
            sample_review,
            inference_model,
            inference_tokenizer,
            inference_device,
            max_length=MAX_LENGTH
        )
        print(f"\nSample Review: \"{prediction['text']}\"")
        print(f"Predicted Sentiment: {prediction['predicted_sentiment']} (Score for Positive: {prediction['prediction_score_positive']:.4f})")

        sample_review_negative = "Terrible movie, I hated every minute of it, boring and poorly acted."
        prediction_negative = predict_sentiment(
            sample_review_negative,
            inference_model,
            inference_tokenizer,
            inference_device,
            max_length=MAX_LENGTH
        )
        print(f"\nSample Review: \"{prediction_negative['text']}\"")
        print(f"Predicted Sentiment: {prediction_negative['predicted_sentiment']} (Score for Positive: {prediction_negative['prediction_score_positive']:.4f})")

    except Exception as e:
        print(f"Error during inference demonstration: {e}")
        print("Make sure you have a model successfully registered in MLflow Model Registry.")


if __name__ == '__main__':
    main()

