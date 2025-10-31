import torch
from improved_transformer import ImprovedDecoderTransformer
import json

model_path = "quran_model.pt"
vocab_path = "vocabulary.json"

with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)

# Try to load the state_dict first to get the keys
state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# Infer d_model from the state_dict if possible
# A common way is to look at the size of the token_embedding weight
# For example, if 'token_embedding.weight' is (vocab_size, d_model)

# Let's assume d_model=800, n_layers=5, n_heads=10, d_ff=3200, dropout=0.1 for now
# If this fails, we'll need a more robust way to infer parameters.

model = ImprovedDecoderTransformer(
    vocab_size=len(vocab),
    d_model=800,
    n_layers=5,
    n_heads=10,
    d_ff=3200,
    dropout=0.1
)

model.load_state_dict(state_dict)

print(model)
