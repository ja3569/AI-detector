import openai
from openai import OpenAI
import os

def generate_text(client, prompt, max_tokens):
    """ Generate text using OpenAI API """
    try:
        # Refer to \https://platform.openai.com/docs/models
        response = client.completions.create(
            model="davinci-002",
            prompt=prompt,
            max_tokens=max_tokens
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return str(e)


def save_text_to_file(folder, filename, text):
    """ Save generated text to a file """
    with open(f"{folder}/{filename}", 'w') as file:
        file.write(text)


def main():
    # Initialize client with OpenAI API Key
    client = OpenAI(
        # Check current usage \https://platform.openai.com/usage
        api_key='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'
    )

    # Prompt users for the number of texts to generate, number of words per text, and number of topics for current set of texts
    n1 = int(input("Enter the number of texts to generate (n1): "))
    n2 = int(input("Enter the number of words per text (n2): "))
    n3 = int(input("Enter the number of topics (n3): "))

    # Prompt users for the n3 number of topics
    topics = [input(f"Enter topic {i+1}: ") for i in range(n3)]
    
    # Create a new folder to store newly-generated training data
    folder_name = 'gptData/' + '_'.join(topics)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Generate texts using GPT 3.0 model and store to the folder
    for i in range(n1):
        prompt = f"Write about {', '.join(topics)}"
        text = generate_text(client, prompt, n2)
        file_name = f"gtext_{i+1}.txt"
        save_text_to_file(folder_name, file_name, text)
        print(f"Text {i+1} saved in folder '{folder_name}' as '{file_name}'")

if __name__ == "__main__":
    main()
