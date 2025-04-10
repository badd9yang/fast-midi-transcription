# Fast MIDI Transcription for Singing Voice Synthesis

This tool provides fast MIDI transcription specifically designed for singing voice synthesis. It relies on F0 extraction tools and the Montreal Forced Aligner (MFA) to identify and extract MIDI information based on F0 trajectories and MFA boundaries. Utilizing a pure post-processing algorithm without dependency on deep learning, this tool achieves a Real-Time Factor (RTF) of approximately 0.02, making it highly competitive compared to existing open-source MIDI transcription tools for singing voice synthesis.

[fig](./demo/out.png)
## Installation

```bash
# Clone the repository
git clone https://github.com/badd9yang/fast-midi-transcription.git
cd fast-midi-transcription

# Install dependencies
pip install -r requirements.txt

# Additional dependencies
# Install Montreal Forced Aligner (MFA)
conda install -c conda-forge montreal-forced-aligner
```

## Usage

### Quick Start

```python
from midi_transcription import MidiTrans

# Example for single-file transcription
wav_file = 'path/to/audio.wav'
textgrid_file = 'path/to/alignment.TextGrid'

transcriber = MidiTrans(sample_rate=44100, hop_length=441, f0_type='pm')
midi_data = transcriber([wav_file, textgrid_file])

# Save MIDI
transcriber.save_midi(midi_data.midi, midi_data.midi_dur, frame_rate=transcriber.frame_rate, midi_path='output.mid')

# Visualization
transcriber.plot(midi_data, save_path='output.png')
```

### Batch Transcription

Supports multi-threaded batch processing:

```python
file_list = [
    ['audio1.wav', 'alignment1.TextGrid'],
    ['audio2.wav', 'alignment2.TextGrid'],
    # More files...
]

transcriber = MidiTrans(file_list=file_list, sample_rate=44100, hop_length=441, f0_type='pm')

# Multi-threaded execution
for midi_data in transcriber.run(max_workers=4):
    print(midi_data.file_name)
```

### Adjusting Transcription Granularity

You can adjust the transcription granularity by setting the `hop_length` parameter. A smaller `hop_length` results in finer granularity, while a larger value provides a coarser transcription:

```python
# Fine-grained transcription
transcriber_fine = MidiTrans(sample_rate=44100, hop_length=220, f0_type='pm')

# Coarse-grained transcription
transcriber_coarse = MidiTrans(sample_rate=44100, hop_length=882, f0_type='pm')
```

## Acknowledgements

This tool references and utilizes the following open-source projects:

- [**ROSVOT**](https://github.com/RickyL-2000/ROSVOT)
- [**So-VITS-SVC**](https://github.com/svc-develop-team/so-vits-svc)
- [**Pseudo-Label Transfer from Frame-level to Note-level in a Teacher-student Framework for Singing Transcription from Polyphonic Music**](https://github.com/keums/icassp2022-vocal-transcription)
## License

It's under the MIT License. It is free for both research and commercial use cases.

