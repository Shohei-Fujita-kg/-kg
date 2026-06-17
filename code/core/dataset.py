from torchaudio.datasets import VCTK_092,TEDLIUM

import core.config as config

def load_dataset(name="ted"):
    """
    データセットを読み込みます \n
    name : "ted","vctk"
    """
    if name=="ted":
        data = TEDLIUM(f"{config.f_path}/data/{config.dataset_path}",release="release1",subset="train",download=True)
    elif name=="vctk":
        data = VCTK_092(f"{config.f_path}/data/{config.dataset_path}",url = 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip',download=True)
    return data