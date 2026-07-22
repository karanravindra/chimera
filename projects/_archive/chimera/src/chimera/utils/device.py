import torch
from typing import Literal


def get_device(device: Literal["auto", "cpu", "cuda", "mps"] = "auto") -> torch.device:
    """Get the device to use for training.

    Args:
        device (Literal["auto", "cpu", "cuda", "mps"], optional): The device to use. Defaults to "auto".

    Returns:
        torch.device: The device to use for training.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")
