
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_REPO = "./Model/"


def load_base_model(args):
    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "revision": 'main',
        "use_auth_token": None,
        "output_hidden_states": True
    }
    config = AutoConfig.from_pretrained(MODEL_REPO + args.model_name, **config_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO + args.model_name,
        config=config,
        torch_dtype=torch.float32,
        device_map='auto',
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO + args.model_name, trust_remote_code=True)
    tokenizer.pad_token_id = 0

    return model, tokenizer, config
