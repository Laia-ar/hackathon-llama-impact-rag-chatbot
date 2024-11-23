from pydub import AudioSegment
from ..config import Config

class AudioLoader:
    @staticmethod
    def load_audio(audio_file_path: str) -> AudioSegment:
        """Carga y normaliza el audio a un formato estándar."""
        audio = AudioSegment.from_file(audio_file_path)
        return audio.set_frame_rate(Config.SAMPLE_RATE)\
                   .set_channels(Config.CHANNELS)\
                   .set_sample_width(Config.SAMPLE_WIDTH)