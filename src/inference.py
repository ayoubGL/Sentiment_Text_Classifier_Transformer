import torch
import mlflow
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_mlflow_model_and_tokenizer(model_uri: str):
    """
    Loads a PyTorch model and tokenizer from MLflow uro
    
    Args:
        Model uri: The MLflow uri
    Returns:
        Tuple: A tuple containing:
            - model : The loaded pytorch model
            - tokenizer: The hugging face tokenizer
            - device: device
    
    """
    print(f"Loading model from MLflow URI:{model_uri}")
    
    try:
        # Load the pytorch model using mlflow.pytorch.load_model
        model = mlflow.pytorch.load_model(model_uri)
        print("Model loaded successfully.")
    except Exception as e:
        print("Error Loading the model")
        raise

    # Determine the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Load the tokenizer form the same MLflow run's artifact
    tokenizer_path = None
    if model_uri.startswith("models:/"):
        print("Attempting to load a tokenizer for inference using model name parameter from MLflow run ...")

        try:
            if "/runs/" in model_uri:
                run_id = model_uri.split("/")[2]
                with mlflow.start_run(run_id=run_id):
                    # check if tokenizer artifact exists
                    client = mlflow.tracking.MlflowClient()
                    artifacts = client.list_artifact(run_id)
                    tokenizer_artifact_found = False
                    for art in artifacts:
                        if art.path == "tokenizer":
                            tokenizer_path = mlflow.artifacts.download_artifacts(
                                run_id = run_id, artifact_path = "tokenizer"
                            )
                            tokenizer_artifact_found = True
                            break
                    if tokenizer_artifact_found:
                        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)       
                        print(f"Tokenizer loaded from run artifacts: {tokenizer_path}")
                    else:
                        # Fallback if tokenizer wasn't logged as a separate artifact or not found
                        print("Tokenizer artifact not found. Falling back to pre-trained tokenizer name")
                        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
                
            else:
                print("Loading tokenizer from pre-trained model name as fallback")
                tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        except Exception as e:
            print(f"Error Loading tokenizer, falling back to 'distilbert-base-uncased': {e}") 
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
              
    return model, tokenizer, device


def predict_sentiment(text: str, model, tokenizer, device: str, max_length: int=128):
    """
        Predict the sentiment for a given text using the loaded model
        
        Args:
            test(str): The input text to predict
            model: The loaded model
            tokenizer: The loading HF tokenizer
            device: The device the mode is on
            max_length: Max sequence length for token
        Return:
            dict: A dic containing predicted_label and prediction score
    """
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()} # Move inputs to device

    with torch.no_grad(): 
        outputs = model(**inputs)
        logits = outputs.logits # get raw predictions
    
    # Apply softmax to get prob
    probabilities = torch.softmax(logits, dim=1)
    # Get the predicted label
    predicted_label = torch.argmax(probabilities, dim=1).item()
    # Get prob for the positive class
    prediction_score = probabilities[0][1].item()
    
    sentiment_map = {0: "Negative", 1: "Positive"}
    
    return {
        "text": text,
        "predicted_label_id": predicted_label,
        "predicted_sentiment": sentiment_map[predicted_label],
        "prediction_score_positive": prediction_score
    }


if __name__ == "__main__":
     print("Please replace 'models:/IMDbSentimentClassifier/1' with your actual MLflow model URI.")
     example_model_uri = "models:'IMDbSentimentClassifier/1"
     try:
        model, tokenizer, device = load_mlflow_model_and_tokenizer(example_model_uri)
        print(f"Model and tokenizer loaded. Model is on device: {device}")
         
        test_reviews = [
            "This movie was absolutely fantastic! I loved every moment of it.",
            "The plot was confusing and the acting was terrible. A total waste of time.",
            "It was an okay film, nothing special but not bad either.",
            "Best movie of the year, a masterpiece!"
        ]

        for review in test_reviews:
            prediction = predict_sentiment(review, model, tokenizer, device, max_length=128)
            print(f"\nReview: \"{prediction['text']}\"")
            print(f"Predicted Sentiment: {prediction['predicted_sentiment']} (Score: {prediction['prediction_score_positive']:.4f})")

     except Exception as e:
        print(f"Could not load model or perform inference. Make sure you have run `main.py` at least once to log a model and that your `example_model_uri` is correct.")
        print(f"Error: {e}")
