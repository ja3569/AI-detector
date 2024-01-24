import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode
import nltk
from nltk.tokenize import word_tokenize
import os
nltk.download('punkt', quiet=True)

class InformationRetrieval(object):
    def __init__(self, jsonKey, engineID):
        self.jk = jsonKey
        self.ed = engineID
        self.oriUrl = f"https://www.googleapis.com/customsearch/v1?key={jsonKey}&cx={engineID}&"

    def retrieveUrls(self, query, total_results=10):
        # Retrieve search result URLs from Google Custom Search API
        urls = []
        start = 1
        while len(urls) < total_results:
            try:
                url = self.oriUrl + urlencode({'q': query, 'num': 10, 'start': start})
                response = requests.get(url)
                response.raise_for_status()
                result = response.json()
                new_urls = [item['link'] for item in result.get('items', [])]
                urls.extend(new_urls)
                start += 10
                if len(new_urls) < 10:
                    break
            except requests.exceptions.RequestException as e:
                print(f"Error making request to Google Custom Search API: {e}")
                break
        return urls[:total_results]

def scrape_webpage(url):
    # Scrape the text content from a webpage URL
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ')
        words = word_tokenize(text)
        return ' '.join(words)
    except requests.exceptions.RequestException:
        return ""

def save_text_to_file(folder, filename, text):
    # Save text content to a file in a specified folder
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(f"{folder}/{filename}", 'w') as file:
        file.write(text)

def generate_combinations(topics, length):
    # Generate combinations of topics for searching
    if length == 0:
        return [[]]
    combinations = []
    for i in range(len(topics)):
        next = topics[i + 1:]
        for rest in generate_combinations(next, length - 1):
            combinations.append([topics[i]] + rest)
    return combinations

def main():
    ir = InformationRetrieval(
        jsonKey="AIzaSyCfTIJgyraAPohYyzkSnPJakFlNEXzD2cE",
        engineID="1148ff606ea12418a"
    )

    n1 = int(input("Enter the number of texts to generate (n1): "))
    n2 = int(input("Enter the number of words per text (n2): "))
    n3 = int(input("Enter the number of topics (n3): "))

    topics = [input(f"Enter topic {i+1}: ") for i in range(n3)]
    base_query = ' '.join(topics)
    folder_name = 'humanData/' + '_'.join(topics)

    urls = ir.retrieveUrls(base_query, total_results=n1)
    file_count = 0

    # Process URLs from the full topic query
    for url in set(urls):
        if file_count >= n1:
            break
        text = scrape_webpage(url)
        if text:
            trimmed_text = ' '.join(text.split()[:n2])
            file_name = f"text_{file_count+1}_full_topic.txt"
            save_text_to_file(folder_name, file_name, trimmed_text)
            print(f"Text {file_count+1} (full topic) saved in folder '{folder_name}' as '{file_name}'")
            file_count += 1

    # If no enough texts, reduce number of topics and search again
    if file_count < n1:
        for i in range(n3-1, 0, -1):
            reduced_topic_combinations = generate_combinations(topics, i)
            for reduced_topics in reduced_topic_combinations:
                if file_count >= n1:
                    break
                reduced_query = ' '.join(reduced_topics)
                additional_urls = ir.retrieveUrls(reduced_query, total_results=1)
                for url in set(additional_urls):
                    if file_count >= n1:
                        break
                    text = scrape_webpage(url)
                    if text:
                        trimmed_text = ' '.join(text.split()[:n2])
                        file_name = f"text_{file_count+1}_reduced_topic.txt"
                        save_text_to_file(folder_name, file_name, trimmed_text)
                        print(f"Text {file_count+1} (reduced topic) saved in folder '{folder_name}' as '{file_name}'")
                        file_count += 1

    if file_count < n1:
        print(f"Only {file_count} texts were generated, which is less than the requested {n1}.")

if __name__ == "__main__":
    main()
