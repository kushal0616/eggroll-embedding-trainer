import torch


def get_device(preferred: str = "auto") -> str:
    if preferred == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return preferred
