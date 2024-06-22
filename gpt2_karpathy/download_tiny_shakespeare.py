import os
import requests

# Define the URL and local file path
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
file_path = "input.txt"

# Download the dataset if it doesn't already exist
if not os.path.exists(file_path):
    response = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(response.content)

# Read the dataset
with open(file_path, 'r') as f:
    text = f.read()

f = open('input.txt', 'w')
f.write(text)
f.close()

# Print the first 1000 characters to verify
# print(text[:1000])
