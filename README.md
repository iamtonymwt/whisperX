<h1 align="center">WhisperX</h1>

### Installation

```bash
pip install whisperx
```


### Usage
```python
model = whisperx.load_model(
  "large-v2", 
  'cuda', 
  device_index=self.device_id, 
  compute_type="float16",**kwargs
)

transcribe_kwargs = dict(
  audio=audio,
  batch_size=self.batch_size,
  language=self.model_kwargs.get('language', None),
  chunk_size=30.0,
  is_music=True, # music flag
  silence_gap=1.0, # max silence gap for music            
  short_segment_threshold=0.5, # minimum segment length for music
)

model.transcribe(**transcribe_kwargs)
```
```python
res = whisperx.assign_word_speakers(
  diarize_segments, 
  res, 
  is_music=is_music # music flag
)
```

### additional logics for music
1. model.transcribe: 3 new args, will go through 'self.vad_model.merge_chunks_music'; Break segments if gap > silence_gap, and eliminate segments < short_segment_threshold.
2. whisperx.assign_word_speakers: 1 new arg, will go through 'assign_word_speakers_music', assign each segment to speakers by calculating the duration of their overlapping speech times from diarization data.
