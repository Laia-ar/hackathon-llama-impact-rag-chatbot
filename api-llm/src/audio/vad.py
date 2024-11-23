import webrtcvad
import collections
from typing import List, Tuple
from pydub import AudioSegment
from ..config import Config

Frame = collections.namedtuple("Frame", ["bytes", "timestamp", "duration"])

class VADProcessor:
    def __init__(self):
        self.vad = webrtcvad.Vad(Config.VAD_AGGRESSIVENESS)
    
    def frame_generator(self, audio_bytes: bytes) -> Frame:
        """Genera frames de audio de duración fija."""
        n = int(Config.SAMPLE_RATE * (Config.FRAME_DURATION_MS / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / (Config.SAMPLE_RATE * 2)) / Config.SAMPLE_RATE
        
        while offset + n <= len(audio_bytes):
            yield Frame(audio_bytes[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n

    def process_audio(self, audio_segment: AudioSegment) -> List[AudioSegment]:
        """Procesa el audio y retorna chunks con voz detectada."""
        frames = list(self.frame_generator(audio_segment.raw_data))
        segments = self._collect_voiced_segments(frames)
        return self._create_audio_chunks(segments, audio_segment)

    def _collect_voiced_segments(self, frames: List[Frame]) -> List[Tuple[float, bytes]]:
        """
        Procesa los frames de audio para detectar segmentos con voz.
        
        Args:
            frames: Lista de frames de audio (Frame namedtuple)
            
        Returns:
            Lista de tuplas (timestamp, bytes) con los segmentos de voz detectados
        """
        num_padding_frames = int(Config.PADDING_DURATION_MS / Config.FRAME_DURATION_MS)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
        voiced_frames = []
        segments = []

        for frame in frames:
            is_speech = self.vad.is_speech(frame.bytes, Config.SAMPLE_RATE)

            # Si aún no se ha detectado voz (no triggered)
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                
                # Si más del 90% de los frames en el buffer son voz, activamos
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    # Agregamos todos los frames del buffer
                    for f, s in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
            
            # Si ya se había detectado voz (triggered)
            else:
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                
                # Si más del 90% de los frames en el buffer son silencio, desactivamos
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    triggered = False
                    # Creamos un segmento con todos los frames de voz acumulados
                    segment = b''.join([f.bytes for f in voiced_frames])
                    segments.append((voiced_frames[0].timestamp, segment))
                    ring_buffer.clear()
                    voiced_frames = []
        
        # Si quedan frames de voz al final, los agregamos como último segmento
        if voiced_frames:
            segment = b''.join([f.bytes for f in voiced_frames])
            segments.append((voiced_frames[0].timestamp, segment))
        
        return segments

    def _create_audio_chunks(self, segments: List[Tuple[float, bytes]], 
                           original_audio: AudioSegment) -> List[AudioSegment]:
        """Convierte segmentos de bytes en AudioSegments y limita su duración."""
        chunks = []
        for _, segment_bytes in segments:
            chunk = AudioSegment(
                data=segment_bytes,
                sample_width=original_audio.sample_width,
                frame_rate=original_audio.frame_rate,
                channels=original_audio.channels
            )
            # Dividir chunks largos en partes más pequeñas
            if len(chunk) > Config.MAX_CHUNK_DURATION_MS:
                chunks.extend(
                    chunk[i:i+Config.MAX_CHUNK_DURATION_MS] 
                    for i in range(0, len(chunk), Config.MAX_CHUNK_DURATION_MS)
                )
            else:
                chunks.append(chunk)
        return chunks