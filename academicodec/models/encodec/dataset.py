import glob
import random
import sys
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

# Add project root to path for shared utilities
_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from datasets.audio_preprocessing import normalize_rms_snr


class NSynthDataset(Dataset):
    """Dataset to load NSynth data."""

    def __init__(self, audio_dir):
        super().__init__()
        self.filenames = []
        self.filenames.extend(glob.glob(audio_dir + "/*.wav"))
        print(len(self.filenames))
        _, self.sr = torchaudio.load(self.filenames[0])
        self.max_len = 24000  # 24000

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        ans = torch.zeros(1, self.max_len)
        audio = torchaudio.load(self.filenames[index])[0]
        
        # Apply RMS/SNR normalization before cropping/padding
        audio = normalize_rms_snr(
            audio,
            target_snr_db=40.0,
            train_mode=True,  # Training mode for HiFiCodec
            snr_variation_db=5.0
        )
        
        if audio.shape[1] > self.max_len:
            st = random.randint(0, audio.shape[1] - self.max_len - 1)
            ed = st + self.max_len
            return audio[:, st:ed]
        else:
            ans[:, :audio.shape[1]] = audio
            return ans
