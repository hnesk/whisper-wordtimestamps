"""
download the models to ./weights
wget https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt -P ./weights
wget https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt -P ./weights
wget https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt -P ./weights
wget https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt -P ./weights
wget https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt  -P ./weights
wget https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt  -P ./weights
"""

import torch
import whisper
import tempfile
import numpy as np
import urllib.request

from typing import Any, Optional
from cog import BasePredictor, Input, Path, BaseModel
from whisper.model import Whisper, ModelDimensions
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE


class ModelOutput(BaseModel):
    detected_language: str
    transcription: str
    segments: Any


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.models = {}

    def get_model(self, model: str) -> Any:
        if not model in whisper.available_models():
            raise Exception("Model %s not found")
        elif not model in self.models:
            self.models[model] = whisper.load_model(
                model, download_root="whisper-cache", device="cuda"
            )
        return self.models[model]

    def predict(
        self,
        audio: Optional[Path] = Input(description="Audio file"),
        audio_url: Optional[str] = Input(description="Audio URL"),
        model: str = Input(
            default="base",
            choices=["tiny", "base", "small", "medium", "large-v1", "large-v2"],
            description="Choose a Whisper model.",
        ),
        language: str = Input(
            choices=sorted(LANGUAGES.keys()),
            default=None,
            description="language spoken in the audio, specify None to perform language detection",
        ),
        temperature: float = Input(
            default=0,
            description="temperature to use for sampling",
        ),
        patience: float = Input(
            default=None,
            description="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search",
        ),
        suppress_tokens: str = Input(
            default="-1",
            description="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations",
        ),
        initial_prompt: str = Input(
            default=None,
            description="optional text to provide as a prompt for the first window.",
        ),
        condition_on_previous_text: bool = Input(
            default=True,
            description="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop",
        ),
        temperature_increment_on_fallback: float = Input(
            default=0.2,
            description="temperature to increase when falling back when the decoding fails to meet either of the thresholds below",
        ),
        compression_ratio_threshold: float = Input(
            default=2.4,
            description="if the gzip compression ratio is higher than this value, treat the decoding as failed",
        ),
        logprob_threshold: float = Input(
            default=-1.0,
            description="if the average log probability is lower than this value, treat the decoding as failed",
        ),
        no_speech_threshold: float = Input(
            default=0.6,
            description="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence",
        ),
        word_timestamps: bool = Input(
            default=False,
            description="Extract word-level timestamps using the cross-attention pattern and dynamic time warping, and include the timestamps for each word in each segment.",
        ),
        prepend_punctuations: str = Input(
            default="\"'“¿([{-",
            description="If word_timestamps is True, merge these punctuation symbols with the next word",
        ),
        append_punctuations: str = Input(
            default="\"'.。,，!！?？:：”)]}、",
            description="If word_timestamps is True, merge these punctuation symbols with the previous word",
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""
        print(f"Transcribe with {model} model")

        model = self.get_model(model).to("cuda")

        if temperature_increment_on_fallback is not None:
            temperature = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperature = [temperature]

        args = {
            "language": language,
            "patience": patience,
            "suppress_tokens": suppress_tokens,
            "initial_prompt": initial_prompt,
            "condition_on_previous_text": condition_on_previous_text,
            "compression_ratio_threshold": compression_ratio_threshold,
            "logprob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "word_timestamps": word_timestamps,
            "prepend_punctuations": prepend_punctuations,
            "append_punctuations": append_punctuations,
        }
        if audio_url is not None:
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                filename = temp_file.name
                http_message, _ = urllib.request.urlretrieve(audio_url, filename)
                if http_message.status == 200:
                    raise Exception("Failed to download audio file")
                result = model.transcribe(filename, temperature=temperature, **args)
        else:
            result = model.transcribe(str(audio), temperature=temperature, **args)

        return ModelOutput(
            segments=result["segments"],
            detected_language=LANGUAGES[result["language"]],
            transcription=result["text"],
        )
