import pickle

# Define a sample dictionary
data = {'name': 'John', 'age': 30, 'city': 'New York'}

# Serialize the dictionary using Pickle
with open('data.pickle', 'wb') as f:
    pickle.dump(data, f)

# Deserialize the dictionary from the Pickle file
with open('data.pickle', 'rb') as f:
    new_data = pickle.load(f)

# Print the deserialized dictionary
print(new_data)