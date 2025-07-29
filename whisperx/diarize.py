import numpy as np
import pandas as pd
from pyannote.audio import Pipeline
from typing import Optional, Union
import torch

from whisperx.audio import load_audio, SAMPLE_RATE
from whisperx.types import TranscriptionResult, AlignedTranscriptionResult


class DiarizationPipeline:
    def __init__(
        self,
        model_name=None,
        use_auth_token=None,
        device: Optional[Union[str, torch.device]] = "cpu",
    ):
        if isinstance(device, str):
            device = torch.device(device)
        model_config = model_name or "pyannote/speaker-diarization-3.1"
        self.model = Pipeline.from_pretrained(model_config, use_auth_token=use_auth_token).to(device)

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_embeddings: bool = False,
    ) -> Union[tuple[pd.DataFrame, Optional[dict[str, list[float]]]], pd.DataFrame]:
        """
        Perform speaker diarization on audio.

        Args:
            audio: Path to audio file or audio array
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect
            return_embeddings: Whether to return speaker embeddings

        Returns:
            If return_embeddings is True:
                Tuple of (diarization dataframe, speaker embeddings dictionary)
            Otherwise:
                Just the diarization dataframe
        """
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }

        if return_embeddings:
            diarization, embeddings = self.model(
                audio_data,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_embeddings=True,
            )
        else:
            diarization = self.model(
                audio_data,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            embeddings = None

        diarize_df = pd.DataFrame(diarization.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)

        if return_embeddings and embeddings is not None:
            speaker_embeddings = {speaker: embeddings[s].tolist() for s, speaker in enumerate(diarization.labels())}
            return diarize_df, speaker_embeddings
        
        # For backwards compatibility
        if return_embeddings:
            return diarize_df, None
        else:
            return diarize_df


def assign_word_speakers(
    diarize_df: pd.DataFrame,
    transcript_result,
    speaker_embeddings: Optional[dict[str, list[float]]] = None,
    fill_nearest: bool = False,
    is_music: bool = False,
) -> Union[AlignedTranscriptionResult, TranscriptionResult]:
    if is_music:
        return assign_word_speakers_music(diarize_df, transcript_result["segments"])
    else:
        return assign_word_speakers_ori(
            diarize_df,
            transcript_result,
            speaker_embeddings=speaker_embeddings,
            fill_nearest=fill_nearest,
        ) 

def assign_word_speakers_music(diarization_df, segments):
    updated_segments = []

    for seg in segments:
        seg_start = seg['start']
        seg_end = seg['end']
        speaker_times = {}

        for _, row in diarization_df.iterrows():
            dia_start = row['start']
            dia_end = row['end']
            speaker = row['speaker']

            overlap_start = max(seg_start, dia_start)
            overlap_end = min(seg_end, dia_end)

            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                speaker_times[speaker] = speaker_times.get(speaker, 0) + overlap_duration

        speaker_list = [(speaker, round(duration, 3)) for speaker, duration in speaker_times.items()]
        seg['speaker'] = speaker_list
        updated_segments.append(seg)

    return updated_segments


def assign_word_speakers_ori(
    diarize_df: pd.DataFrame,
    transcript_result,
    speaker_embeddings: Optional[dict[str, list[float]]] = None,
    fill_nearest: bool = False,
) -> Union[AlignedTranscriptionResult, TranscriptionResult]:
    """
    Assign speakers to words and segments in the transcript.

    Args:
        diarize_df: Diarization dataframe from DiarizationPipeline
        transcript_result: Transcription result to augment with speaker labels
        speaker_embeddings: Optional dictionary mapping speaker IDs to embedding vectors
        fill_nearest: If True, assign speakers even when there's no direct time overlap

    Returns:
        Updated transcript_result with speaker assignments and optionally embeddings
    """
    transcript_segments = transcript_result["segments"]
    for seg in transcript_segments:
        # assign speaker to segment (if any)
        diarize_df['intersection'] = np.minimum(diarize_df['end'], seg['end']) - np.maximum(diarize_df['start'], seg['start'])
        diarize_df['union'] = np.maximum(diarize_df['end'], seg['end']) - np.minimum(diarize_df['start'], seg['start'])
        # remove no hit, otherwise we look for closest (even negative intersection...)
        if not fill_nearest:
            dia_tmp = diarize_df[diarize_df['intersection'] > 0]
        else:
            dia_tmp = diarize_df
        if len(dia_tmp) > 0:
            # sum over speakers
            speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
            seg["speaker"] = speaker
        
        # assign speaker to words
        if 'words' in seg:
            for word in seg['words']:
                if 'start' in word:
                    diarize_df['intersection'] = np.minimum(diarize_df['end'], word['end']) - np.maximum(diarize_df['start'], word['start'])
                    diarize_df['union'] = np.maximum(diarize_df['end'], word['end']) - np.minimum(diarize_df['start'], word['start'])
                    # remove no hit
                    if not fill_nearest:
                        dia_tmp = diarize_df[diarize_df['intersection'] > 0]
                    else:
                        dia_tmp = diarize_df
                    if len(dia_tmp) > 0:
                        # sum over speakers
                        speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
                        word["speaker"] = speaker

    # Add speaker embeddings to the result if provided
    if speaker_embeddings is not None:
        transcript_result["speaker_embeddings"] = speaker_embeddings

    return transcript_result


class Segment:
    def __init__(self, start:int, end:int, speaker:Optional[str]=None):
        self.start = start
        self.end = end
        self.speaker = speaker
