import random
import numpy as np
import torch
from torch.nn import functional as F
import frozendict
import seaborn as sns
import ml_collections

def get_publaynet_config():
	"""Gets the default hyperparameter configuration."""

	config = ml_collections.ConfigDict()
	# Exp info
	config.name = 'Publaynet'
	config.ckpt_path = None
	config.dataset_path = "data/publaynet"
	config.limit = 20

	# Training info
	config.epoch = 100
	config.batch_size = 64
	config.train_shuffle = True
	config.eval_pad_last_batch = False
	config.eval_batch_size = 64
	config.save_every_epoch = 25

	# Model info
	config.num_layers = 6
	config.qkv_dim = 512
	config.num_heads = 8

	# Optimizer info
	config.optimizer = ml_collections.ConfigDict()
	config.optimizer.lr = 1e-4
	config.optimizer.beta1 = 0.9
	config.optimizer.beta2 = 0.95
	config.optimizer.weight_decay = 0.1
	config.optimizer.grad_norm_clip = 1.0
	config.optimizer.lr_decay = True
	config.optimizer.warmup_iters = 2000
	config.optimizer.final_iters = 10000

	# Dataset info
	config.dataset = ml_collections.ConfigDict()
	config.dataset.LABEL_NAMES = ("text", "title", "list", "table", "figure")
	config.dataset.COLORS = {
			"title": (193, 0, 0),
			"list": (64, 44, 105),
			"figure": (36, 234, 5),
			"table":  (89, 130, 213),
			"text": (253, 141, 28),
			"background": (200, 200, 200)}

	config.dataset.FRAME_WIDTH = 1050
	config.dataset.FRAME_HEIGHT = 1485

	config.dataset.ID_TO_LABEL = frozendict.frozendict({
			0: "text",
			1: "title",
			2: "list",
			3: "table",
			4: "figure",
		})
	config.dataset.NUMBER_LABELS = 5
	config.dataset.LABEL_TO_ID = frozendict.frozendict(
			{l: i for i,l in config.dataset.ID_TO_LABEL.items()})

	return config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def is_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            raise ImportError("console")
            return False
    except:
        return False
    else:  # pragma: no cover
        return True

def gen_colors(num_colors):
    """
    Generate uniformly distributed `num_colors` colors
    :param num_colors:
    :return:
    """
    palette = sns.color_palette(None, num_colors)
    rgb_triples = [[int(x[0]*255), int(x[1]*255), int(x[2]*255)] for x in palette]
    return rgb_triples


@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.module.get_block_size() if hasattr(model, "module") else model.getcond_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x

def trim_tokens(tokens, bos, eos, pad=None):
    bos_idx = np.where(tokens == bos)[0]
    tokens = tokens[bos_idx[0]+1:] if len(bos_idx) > 0 else tokens
    eos_idx = np.where(tokens == eos)[0]
    tokens = tokens[:eos_idx[0]] if len(eos_idx) > 0 else tokens
    # tokens = tokens[tokens != bos]
    # tokens = tokens[tokens != eos]
    if pad is not None:
        tokens = tokens[tokens != pad]
    return tokens