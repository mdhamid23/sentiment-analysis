from telegram_bot_provider import TelegramBotProvider
import asyncio
import sys

def main():
    bot_provider = TelegramBotProvider()
    bot_application = bot_provider.get_bot_application()

    # Get the event loop, create if necessary
    loop = asyncio.get_event_loop()

    # Initialize the bot application
    loop.run_until_complete(bot_application.initialize())

    # Initialize handlers
    bot_provider.initialize_handlers()

    print("Starting polling...")
    loop.run_until_complete(bot_application.run_polling())

    # Gracefully shut down the bot
    loop.run_until_complete(bot_application.shutdown())

if __name__ == "__main__":
    try:
        print("Starting new event loop...")
        main()
    except RuntimeError as e:
        print(f"Error: {e}")
