import json
import random
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import transformers


TASKS = {'taxi_pick': 266, 'taxi_drop': 266, 'bike_pick': 250, 'bike_drop': 250}
SEEDS = (2026, 2027, 2028, 2029, 2030)
ROOT = Path(__file__).resolve().parents[1]


def protocol():
    return json.loads((ROOT / 'configs' / 'table2.json').read_text(encoding='utf-8'))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment(device):
    return {'python': platform.python_version(), 'interpreter': sys.executable, 'platform': platform.platform(),
            'torch': torch.__version__, 'transformers': transformers.__version__, 'numpy': np.__version__,
            'cuda_build': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'device': str(device),
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark}
