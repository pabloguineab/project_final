import pickle

# Load the contents of the file
with open('train_validate_features1.pkl', 'rb') as file:
    contents = pickle.load(file)

# Inspect the contents
print(type(contents))
if isinstance(contents, dict):
    # Likely to be image features; check a couple of keys
    print(list(contents.keys())[:5])
elif isinstance(contents, Tokenizer):
    # If it's a tokenizer, print part of the word index
    print(list(contents.word_index.items())[:5])
else:
    print("Unknown content type.")