import json
import random

# Sample messages for non-null values
sample_messages = [
    "Free entry in a weekly competition to win FA Cup final tkts",
    "Hey, what are you doing today?",
    "Congratulations you've won a prize! Reply with WIN to claim.",
    "Don't forget to call me when you have time.",
    "Get your exclusive deal now! Offer ends today.",
    "This is a reminder for your appointment tomorrow.",
    "You have an unread message waiting for you.",
    "Limited time offer: Buy one get one free!",
    "Update your account information to avoid suspension.",
    "Your delivery is scheduled for tomorrow.",
    "Your package is ready for pickup.",
    "Meeting rescheduled to 3 PM.",
    "Hello, how are you?",
    "Reminder: Your subscription is expiring soon.",
    "Can you send me the details by email?",
    "Your account balance is low. Recharge now.",
    "Join us for a free webinar this Friday.",
    "Don't miss out on this amazing opportunity!",
    "Call us back at 123-456-7890 for more details.",
    "Verify your email to activate your account."
]

# Generate 100 messages with some null values
messages = []
for i in range(1, 101):
    value = random.choice(sample_messages) if random.random() > 0.1 else None
    messages.append({
        "key": f"msg{i}",
        "value": value
    })

# Save to JSON
input_data_path = 'input_data.json'
with open(input_data_path, 'w') as f:
    json.dump(messages, f, indent=2)

input_data_path
