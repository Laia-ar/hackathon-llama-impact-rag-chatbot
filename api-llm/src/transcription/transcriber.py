import io
from typing import List
from openai import OpenAI
from pydub import AudioSegment
import whisper

class Transcriber:

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
            model = whisper.load_model("tiny")
            transcript = model.transcribe(wav_file)["text"]
        return transcript.text