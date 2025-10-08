import sys
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

espeak_lib_path = r"C:\Users\franc\scoop\apps\espeak-ng\current\espeak NG\libespeak-ng.dll"
espeak_data_path = r"C:\Users\franc\scoop\apps\espeak-ng\current\espeak NG\espeak-ng-data"

def configure_espeak_windows():
    if not os.path.exists(espeak_lib_path):
        print(f"Warning: eSpeak NG library not found at {espeak_lib_path}")
    else:
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeak_lib_path

    if not os.path.exists(espeak_data_path):
        print(f"Warning: eSpeak NG data folder not found at {espeak_data_path}")
    else:
        os.environ["ESPEAK_DATA_PATH"] = espeak_data_path

if sys.platform == "win32":
    configure_espeak_windows()

model_name = "facebook/wav2vec2-lv-60-espeak-cv-ft"
Wav2Vec2Processor.from_pretrained(model_name).save_pretrained("./wav2vec2_model")
Wav2Vec2ForCTC.from_pretrained(model_name).save_pretrained("./wav2vec2_model")