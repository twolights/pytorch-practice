import sys
from typing import Optional

import torch


def device(device_type: Optional[str] = None):
    if device_type is None:
        try:
            device_type = sys.argv[1]
        except IndexError:
            print("Device type not specified, using CPU")
            device_type = "cpu"
    return torch.device(device_type)
