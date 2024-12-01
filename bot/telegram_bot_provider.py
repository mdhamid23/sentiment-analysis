from telegram.ext import Application, MessageHandler, filters,CommandHandler
import credentials as cred
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModel,AdamW, get_scheduler, pipeline
import torch
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import classification_report
from langdetect import detect

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
        # Add command handlers
        self.bot_application.add_handler(CommandHandler("menu", self.menu_command))
        self.bot_application.add_handler(CommandHandler("help", self.help_command))
        self.bot_application.add_handler(CommandHandler("about", self.about_command))
        self.bot_application.add_handler(CommandHandler("config", self.config_command))

        # Add message handler for non-command messages
        message_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, self.on_message)
        self.bot_application.add_handler(message_handler)

        # Error handler (optional)
        self.bot_application.add_error_handler(self.handle_polling_error)
    

    async def menu_command(self, update, context):
        """Handle /menu command."""
        response = (
            "<b>📋 Main Menu</b>\n\n"
            "👋 Welcome to the bot! Choose an option below:\n\n"
            "🔹 <b>/help</b> - Get assistance on how to use the bot 🆘\n"
            "🔹 <b>/config</b> - View bot configurations ⚙️\n"
            "🔹 <b>/about</b> - Learn more about this bot 🤖"
        )
        await update.message.reply_text(response, parse_mode='HTML', disable_web_page_preview=True)

    async def help_command(self, update, context):
        """Handle /help command."""
        response = (
            "<b>🆘 Help Guide</b>\n\n"
            "💬 Simply send me a message in either:\n"
            "   - Bangla 🇧🇩\n"
            "   - English 🇬🇧\n\n"
            "📊 I'll analyze the sentiment for you and reply with:\n"
            "   - Positive 😊\n"
            "   - Neutral 😐\n"
            "   - Negative 😞\n\n"
            "Feel free to experiment and explore!"
        )
        await update.message.reply_text(response, parse_mode='HTML', disable_web_page_preview=True)

    async def about_command(self, update, context):
        """Handle /about command."""
        response = (
            "<b>🤖 About This Bot</b>\n\n"
            "🌟 This bot is designed to analyze sentiments in both <b>English</b> and <b>Bangla</b> text.\n\n"
            "🛠️ Built using:\n"
            "   - <b>Python</b> 🐍\n"
            "   - <b>Pre-trained Language Model</b> 🤗\n"
            "   - <b>Telegram Bot API</b> 📡\n\n"
            "🚀 Feel free to explore its features and share your thoughts!"
        )
        await update.message.reply_text(response, parse_mode='HTML', disable_web_page_preview=True)


    async def config_command(self, update, context):
        """Handle /config command."""
        response = (
            "<b><i>⚙️ Bot Configurations</i></b>\n\n"
            "🌟 Welcome to the Bilingual Sentiment Analysis Bot! 🌟\n\n"
            "🔍 This bot analyzes sentiments in both <b>English</b> and <b>Bangla</b> text.\n\n"
            "🌐 <b>Bangla Sentiment Analysis</b>\n"
            "   - Powered by <b>sagorsarker/bangla-bert-base</b> 🛠️\n"
            "   - Fine-tuned with a Bangla sentiment dataset 📊\n\n"
            "🌐 <b>English Sentiment Analysis</b>\n"
            "   - Uses <b>distilbert-base-uncased-finetuned-sst-2-english</b> 🛠️\n"
            "   - Delivers state-of-the-art results 🌟\n\n"
            "✨ Enjoy using the bot and let us know your feedback! 📝"
        )
        await update.message.reply_text(response, parse_mode='HTML', disable_web_page_preview=True)


    async def on_message(self, update, context):
        """Handle incoming messages."""
        processing_message = await update.message.reply_text(
        "⏳ Processing your message, please wait...",
        parse_mode="HTML"
    )
        response = ''
        lang_code = detect(update.message.text)
        print("Detected language: ", lang_code)
        if lang_code == "bn":  # Bangla
            sentiment = await self.process_bangla_text(update.message.text)
            response = "The Text Seems "+sentiment  + "!"
        else:
            sentiment = await self.process_english_text(update.message.text)
            response = "The Text Seems "+sentiment  + "!"
        
        #! Delete the "processing" message
        await context.bot.delete_message(
            chat_id=update.message.chat_id,
            message_id=processing_message.message_id
        )
        
        # await update.message.reply_text(response)
        await update.message.reply_text(
        response,
        parse_mode="HTML",
        disable_web_page_preview=True,
        reply_to_message_id=update.message.message_id  # Reference the original message
    )

        #! Only run for training the bangla model
        # self.fine_tuning_model()

    async def handle_polling_error(self, update, context):
        """Handle errors."""
        print(f"Polling error: {context.error}")


    async def process_bangla_text(self, text):
        # Load the trained model and tokenizer
        model_path = "bangla-sentiment-model"  # Replace with your folder path
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Move model to the appropriate device (GPU or CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        sentiment = self.predict_sentiment(text, model, tokenizer, device)
        
        return sentiment

    async def process_english_text(self, text):
        # Load the sentiment analysis pipeline
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        result = sentiment_analyzer(text)
        print(f"Text: {text}\nSentiment: {result[0]['label']}, Score: {result[0]['score']}\n")
        sentiment = ''
        if result[0]['label'] == 'POSITIVE':
            sentiment = 'Positive'
        elif result[0]['label'] == 'NEGATIVE':
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        return sentiment

    def predict_sentiment(self,text, model, tokenizer, device):
        # Tokenize the input text
        encoding = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)  # Move tensors to the same device as the model

        # Perform inference
        model.eval()
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()  # Get the predicted class index

        # Map the prediction to sentiment
        sentiment_map = {0: "Neutral", 1: "Positive", 2: "Negative"}
        return sentiment_map[prediction]

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
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3,device_map = "auto")


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
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4)
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
            print('training...')
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
