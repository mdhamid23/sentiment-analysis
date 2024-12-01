from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from telegram.error import Forbidden
import logging

class BotService:
    def __init__(self, bot_application: Application):
        self.bot_application = bot_application

    async def on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Handle incoming messages
        message = update.message
        if message:
            chat_id = message.chat_id
            await context.bot.send_message(chat_id=chat_id, text="The bot is currently under maintenance. Please try again later.")

    async def handle_polling_error(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            raise context.error
        except Forbidden as e:
            logging.error("Bot was removed or kicked from the group. Please re-add the bot.")
        except Exception as e:
            logging.error(f"An error occurred: {e}")

    def initialize_handlers(self):
        print("inside intialize_handlers")
        # Message handler
        self.bot_application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.on_message))

        # Polling error handler
        self.bot_application.add_error_handler(self.handle_polling_error)
