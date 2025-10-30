import torch, random, numpy as np, os

def set_seed(seed: int):
    set_hash_seed(seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_hash_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)

def enable_determinism():
    torch.use_deterministic_algorithms(True)

def worker_init_fn(worker_id):
    def init_fn(_):
        np.random.seed(torch.initial_seed() % (2**32) + worker_id)
    return init_fn