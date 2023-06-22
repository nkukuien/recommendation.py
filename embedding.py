import csv
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Read college data from the CSV file
colleges = []
with open('/content/Embedings.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        colleges.append(row)

# Encode the descriptions and generate BERT embeddings
embeddings = []
for college in colleges:
    description = college['Descriptions']
    # Tokenize the description
    tokens = tokenizer.encode(description, add_special_tokens=True)
    # Convert tokens to tensors
    input_ids = torch.tensor(tokens).unsqueeze(0)
    # Generate BERT embeddings
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

# Reshape the embeddings
embeddings = torch.tensor(embeddings)

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# Now, suppose a user likes "Arusha Technical College (ATC) - Arusha". We can recommend another college based on cosine similarity.
liked_college = "Arusha Technical College (ATC) - Arusha"
liked_college_index = next(index for index, college in enumerate(colleges) if college['colleges'] == liked_college)

# Find the most similar college
similar_college_index = similarity_matrix[liked_college_index].argsort()[::-1][1]  # Exclude the liked college itself
recommended_college = colleges[similar_college_index]

print("Because you liked " + liked_college + ", we recommend: " + recommended_college['colleges'
