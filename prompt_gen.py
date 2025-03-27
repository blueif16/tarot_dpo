import random
from datasets import Dataset
from huggingface_hub import login

from dotenv import load_dotenv
import os

load_dotenv()

# Set a random seed for reproducibility (optional, remove if you want different prompts each time)
random.seed(42)

# Define card_names (as provided in your query)
card_names = [
    'The Fool', 'The Magician', 'The High Priestess', 'The Empress', 'The Emperor',
    'The Hierophant', 'The Lovers', 'The Chariot', 'Strength', 'The Hermit',
    'Wheel of Fortune', 'Justice', 'The Hanged Man', 'Death', 'Temperance',
    'The Devil', 'The Tower', 'The Star', 'The Moon', 'The Sun', 'Judgment',
    'The World', 'Ace of Wands', 'Two of Wands', 'Three of Wands', 'Four of Wands',
    'Five of Wands', 'Six of Wands', 'Seven of Wands', 'Eight of Wands', 'Nine of Wands',
    'Ten of Wands', 'Page of Wands', 'Knight of Wands', 'Queen of Wands', 'King of Wands',
    'Ace of Cups', 'Two of Cups', 'Three of Cups', 'Four of Cups', 'Five of Cups',
    'Six of Cups', 'Seven of Cups', 'Eight of Cups', 'Nine of Cups', 'Ten of Cups',
    'Page of Cups', 'Knight of Cups', 'Queen of Cups', 'King of Cups', 'Ace of Swords',
    'Two of Swords', 'Three of Swords', 'Four of Swords', 'Five of Swords', 'Six of Swords',
    'Seven of Swords', 'Eight of Swords', 'Nine of Swords', 'Ten of Swords', 'Page of Swords',
    'Knight of Swords', 'Queen of Swords', 'King of Swords', 'Ace of Pentacles', 'Two of Pentacles',
    'Three of Pentacles', 'Four of Pentacles', 'Five of Pentacles', 'Six of Pentacles',
    'Seven of Pentacles', 'Eight of Pentacles', 'Nine of Pentacles', 'Ten of Pentacles',
    'Page of Pentacles', 'Knight of Pentacles', 'Queen of Pentacles', 'King of Pentacles',
]

# Define topics (as provided in your query)
topics = {
    "Relationships": ["their mind", "relationship flow", "your move", "challenge", "solution", "evolution"],
    "Wealth": ["current", "future", "opportunity", "resource", "action", "reward"],
    "Career": ["now", "challenge", "future", "skill", "potential", "opportunity"],
    "Self-Growth": ["now", "block", "potential", "mind", "spirit", "guide"],
    "Health": ["past", "present", "future", "challenge", "advice"],
    "Spiritual Guidance": ["past", "present", "future", "challenge", "advice"],
    "Life Purpose": ["past", "present", "future", "challenge", "advice"],
    "Daily Guidance": ["event", "emotion", "reflection"],
    "Weekly Guidance": ["event", "emotion", "reflection"]
}

main_topics = ["Relationships", "Wealth", "Career", "Self-Growth", "Health", "Spiritual Guidance", "Life Purpose"]
guidance_topics = ["Daily Guidance", "Weekly Guidance"]

# Function to generate prompt for main topics
def generate_main_prompt(card, topic, positional_meanings):
    pos_meaning = random.choice(positional_meanings)
    return f"What does {card} reveal about {pos_meaning} in my {topic}?"

# Function to generate prompts for guidance topics
def generate_guidance_prompts(card, topic):
    prompts = []
    for pos_meaning in topics[topic]:
        prompts.append(f"What {pos_meaning} does {card} suggest for my {topic.lower()}?")
    return prompts

# Generate all prompts
prompts = []
# Fix: Use card_names instead of tarot_cards (typo in your original script)
for card in card_names:
    # Generate prompts for main topics (1 prompt per topic per card)
    for topic in main_topics:
        prompt = generate_main_prompt(card, topic, topics[topic])
        prompts.append({"prompt": prompt})  # Store as dictionary for Dataset
    # Generate prompts for guidance topics (3 prompts per topic per card)
    for topic in guidance_topics:
        guidance_prompts = generate_guidance_prompts(card, topic)
        for gp in guidance_prompts:
            prompts.append({"prompt": gp})  # Store as dictionary for Dataset

# Create a dataset from the list of prompts
dataset = Dataset.from_list(prompts)

# Login to Hugging Face (run `huggingface-cli login` in terminal or set token below)
# Uncomment and set your token if not using CLI: login(token="your_token_here")
from huggingface_hub import login
login()

# Upload the dataset to Hugging Face
# Replace "your_username" with your Hugging Face username and choose a dataset name
dataset.push_to_hub(os.getenv("REPO_ID"))

print("Dataset uploaded successfully.")
print(f"Total prompts generated: {len(prompts)}")
print("\nSample prompts:")
for prompt in prompts[:5]:
    print(f"- {prompt['prompt']}")