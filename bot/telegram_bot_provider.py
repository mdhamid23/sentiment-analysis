from telegram.ext import Application, MessageHandler, filters
import credentials as cred
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModel,AdamW, get_scheduler
import torch
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import classification_report

class TelegramBotProvider:
    def __init__(self):
        print("Initializing TelegramBotProvider...")
        token = cred.TELEGRAM_BOT_TOKEN
        print(f"Token: {token}")
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is not defined in credentials.")
        self.bot_application = Application.builder().token(token).build()

    def get_bot_application(self):
        return self.bot_application

    def initialize_handlers(self):
        print("Inside initialize_handlers...")

        # Example message handler
        message_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, self.on_message)
        self.bot_application.add_handler(message_handler)

        # Error handler (optional)
        self.bot_application.add_error_handler(self.handle_polling_error)

    async def on_message(self, update, context):
        """Handle incoming messages."""
        response = await self.process_text(update.message.text)
        await update.message.reply_text(response)
        self.fine_tuning_model()

    async def handle_polling_error(self, update, context):
        """Handle errors."""
        print(f"Polling error: {context.error}")



    async def process_text(self, text):
         # Step 4: Load the Pre-trained Model for Sentiment Analysis
        # Load tokenizer and model
        model_name = "sagorsarker/bangla-bert-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        # Step 5: Create a Sentiment Analysis Pipeline
        sentences = []
        sentences.append(text)
        # sentences = ["আমি খুবই খুশি।", "আজকের দিনটি ভালো ছিল না।"]
        print("the sentence is: ", sentences)
        tokens = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors="pt")

        predictions = None
        with torch.no_grad():
            predictions = model(**tokens)
            # predictions = torch.argmax(outputs.logits, dim=-1)
        embeddings = predictions.last_hidden_state
        print("the embeddings are: ", embeddings)
        print(embeddings.shape)  # Shape: (batch_size, sequence_length, hidden_si
        # print("the predictions are: ", predictions)
        cls_embeddings = embeddings[:, 0, :]  # [batch_size, hidden_size]

        # Placeholder for simple sentiment logic
        # (You'd replace this with an actual classifier if needed.)
        sentiment_scores = torch.rand((len(sentences), 3))  # Simulating output scores for 3 classes
        sentiments = torch.argmax(F.softmax(sentiment_scores, dim=1), dim=1)

        # Map predictions to sentiment labels
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        predicted_sentiments = [sentiment_map[s.item()] for s in sentiments]

        for sentence, sentiment in zip(sentences, predicted_sentiments):
            print(f"Sentence: {sentence}\nPredicted Sentiment: {sentiment}\n")

        return "The Text Seems "+sentiment  + "!"


    def fine_tuning_model(self):
        data_path = "Train.csv"  # Replace with the path to your dataset
        df = pd.read_csv(data_path)

        # Inspect dataset
        print(df.head())
        # Check for missing values
        print(df.isnull().sum())
        # Drop or handle missing values
        df = df.dropna()

        # Ensure columns are named "Data" and "Label"
        assert "Data" in df.columns and "Label" in df.columns, "Dataset must have 'Data' and 'Label' columns"

        # Check label distribution
        print(df["Label"].value_counts())

        # ! 3. Preprocess and Split the Data
        # *  Split dataset into training and validation sets
        # ? blue comment
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df["Data"].values, df["Label"].values, test_size=0.2, random_state=42
        )

        print("end of process 3...")
        #! 4. Load the Pre-trained Model and Tokenizer
        # *Use the sagorsarker/bangla-bert-base model and its tokenizer.

        model_name = "sagorsarker/bangla-bert-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model with a classification head for 3 labels
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)


        print("end of process 4...")
        #! 5. Tokenize the Dataset
        def tokenize_data(texts, tokenizer, max_length=128):
            return tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        train_encodings = tokenize_data(train_texts, tokenizer)
        val_encodings = tokenize_data(val_texts, tokenizer)
        print("end of process 5...")

        #! 6. Create a PyTorch Dataset
        #* Wrap the tokenized data into PyTorch datasets.
        class SentimentDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item
        
        # Create PyTorch datasets
        train_dataset = SentimentDataset(train_encodings, train_labels)
        val_dataset = SentimentDataset(val_encodings, val_labels)

        print("end of process 6...")

        #! 7. Define Training Parameters
        #* Set up training arguments and the optimizer.
        # Optimizer
        optimizer = AdamW(model.parameters(), lr=2e-5)

        # Scheduler
        num_epochs = 3
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16)
        num_training_steps = len(train_dataloader) * num_epochs
        lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        print("end of process 7...")

        #! 8. Train the Model
        #* Train the model using PyTorch.
                                
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Training loop
        loss_fn = CrossEntropyLoss()

        for epoch in range(num_epochs):
            model.train()
            loop = tqdm(train_dataloader, leave=True)
            for batch in loop:
                # Move data to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                # Print metrics
                loop.set_description(f"Epoch {epoch}")
                loop.set_postfix(loss=loss.item())

        # Save the model
        model.save_pretrained("./bangla-sentiment-model")
        tokenizer.save_pretrained("./bangla-sentiment-model")

        print("end of process 8...")

        #! 9. Evaluate the Model
        #*Evaluate the model on the validation set.
        model.eval()
        predictions, true_labels = [], []

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)

                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # Classification report
        print(classification_report(true_labels, predictions, target_names=["Neutral", "Positive", "Negative"]))
        print("end of process 9...")
