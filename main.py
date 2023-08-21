from src.utils import load_encoder_hparams_and_params
from src.gpt2 import generate

prompt = "Hello there?"
n_tokens_to_generate = it = 100
model_size = str = "774M"
models_dir = str = "models"


# load encoder, hparams, and params from the released open-ai gpt-2 files
encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

# encode the input string using the BPE tokenizer
input_ids = encoder.encode(prompt)

# make sure we are not surpassing the max sequence length of our model
assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

# generate output ids
output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)

# decode the ids back into a string
output_text = encoder.decode(output_ids)

print(output_text)
