import json

fisher_dict = {'a':1}
optpar_dict = {'b':2}
# Save dictionary to JSON file
file_path = 'fisher_dict.json'
with open(file_path, 'w') as f:
    json.dump(fisher_dict, f)

file_path = 'optpar_dict.json'
with open(file_path, 'w') as f:
    json.dump(optpar_dict, f)

print("Dictionaries saved successfully!")