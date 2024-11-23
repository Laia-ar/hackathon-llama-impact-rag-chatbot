import io
from typing import List
from openai import OpenAI
from pydub import AudioSegment
import requests
from ..config import Config

class Transcriber:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)

    def transcribe_chunks(self, audio_chunks: List[AudioSegment]) -> str:
        """Transcribe una lista de chunks de audio."""
        transcriptions = []
        for chunk in audio_chunks:
            transcription = self._transcribe_single_chunk(chunk)
            transcriptions.append(transcription)
        return ' '.join(transcriptions)

    def _transcribe_single_chunk(self, chunk: AudioSegment) -> str:
        """Transcribe un único chunk de audio."""
        with io.BytesIO() as wav_file:
            chunk.export(wav_file, format="wav")
            wav_file.seek(0)
            wav_file.name = "audio.wav"
            transcript = requests.post(
                 "https://api.aimlapi.com/stt",
                 headers={"Content-Type":"application/json"},
                 json={"model":"text"}
                       )
            transcript = transcript.json()
        return transcript.text