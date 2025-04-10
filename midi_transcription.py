# Copyright by YangChen [2025/4/9]

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
import concurrent.futures
import threading

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import torch
import torch.nn.functional as F
import torchaudio
from scipy.interpolate import interp1d
from textgrid import TextGrid
from torch import Tensor
from tqdm import tqdm

import F0_predictor
from midi_utils import get_midi


@dataclass
class MidiData:
    """A dataclass for storing all MIDI-related information.
    Compared to dictionaries, it allows accessing elements via `A.x` syntax,
    prevents adding invalid elements, and avoids typos.
    
    Usage:
        [Writing methods]
        >>> data = MidiData()       # Initialize
        
        # 1. Direct assignment
        >>> data.file_name = "<Song_name>"
        >>> data.midi = [56,57,59]
        >>> ... 
        >>> data.midi
        [56,57,59]
        
        # 2. From dictionary
        >>> midi_dict = {"file_name": "<Song_name>",
        ...               "midi": [56,57,59],
        ...                ... }
        >>> data = MidiData()
        >>> data.from_dict(midi_dict)
        
        # 3. From existing MidiData
        >>> data = MidiData()
        >>> data2 = MidiData()
        >>> data.file_name = "<Song_name>"
        >>> data.midi = [56,57,59]
        >>> data2.from_data(data)
    """

    file_name: Optional[Union[List[str], str]] = None
    ph: Optional[Union[Tensor, np.ndarray, list]] = None            # Phonemes
    ph_dur: Optional[Union[Tensor, np.ndarray, list]] = None        # Phoneme durations
    word: Optional[Union[Tensor, np.ndarray, list]] = None          # Words
    word_dur: Optional[Union[Tensor, np.ndarray, list]] = None      # Word durations
    ph2word: Optional[Union[Tensor, np.ndarray, list]] = None       # Phoneme-to-word mapping

    midi: Optional[Union[Tensor, np.ndarray, list]] = None          # NOTE: Word-level MIDI
    midi_dur: Optional[Union[Tensor, np.ndarray, list]] = None      # Word-level MIDI durations
    midi_len: Optional[Union[Tensor, np.ndarray, list]] = None      # Word-level MIDI length
    midi2word: Optional[Union[Tensor, np.ndarray, list]] = None     # MIDI-to-word mapping

    midi_ph: Optional[Union[Tensor, np.ndarray, list]] = None       # NOTE: Phoneme-level MIDI
    midi_ph_dur: Optional[Union[Tensor, np.ndarray, list]] = None   # Phoneme-level MIDI durations
    midi_ph_len: Optional[Union[Tensor, np.ndarray, list]] = None   # Phoneme-level MIDI length
    midi2ph: Optional[Union[Tensor, np.ndarray, list]] = None       # MIDI-to-phoneme mapping

    midi_frame: Optional[Union[Tensor, np.ndarray, list]] = None    # NOTE: Frame-level MIDI

    f0: Optional[Union[Tensor, np.ndarray, list]] = None            # Frame-level F0
    audio: Optional[Union[Tensor, np.ndarray]] = None               # Audio waveform

    # Can add other elements as needed

    def __getitem__(self, key: str) -> Any:
        """Retrieve an attribute by its name."""
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found.")

    def __setitem__(self, key: str, value: Any):
        """Set an attribute by its name."""
        if key in self.__dataclass_fields__:
            setattr(self, key, value)
        else:
            raise KeyError(f"Key '{key}' not found.")

    def cuda(self, rank: int = 0, non_blocking: bool = False):
        """Move all tensor attributes to the specified CUDA device."""
        for field_info in fields(self):
            attr_value = getattr(self, field_info.name)
            if isinstance(attr_value, torch.Tensor):
                setattr(self, field_info.name, attr_value.cuda(rank, non_blocking))
            elif isinstance(attr_value, list):
                # Assume the list contains tensors
                new_list = []
                for item in attr_value:
                    if isinstance(item, torch.Tensor):
                        new_list.append(item.cuda(rank, non_blocking))
                    else:
                        new_list.append(item)
                setattr(self, field_info.name, new_list)

    def get_value(self, key: str, default: Any = None) -> Any:
        """Return the value of the specified attribute.
        If the attribute does not exist or its value is None,
        return `default`.
        """
        return (
            getattr(self, key, default)
            if getattr(self, key, default) is not None
            else default
        )

    def update(self, other: Union[Dict[str, Any], "MidiData"]) -> "MidiData":
        """Update this object with another MidiData or a dictionary and return a new instance."""
        if isinstance(other, MidiData):
            other = other.to_dict()
        new_data = self.to_dict()
        new_data.update(other)
        return self.from_dict(new_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the object into a dictionary."""
        return {
            field_info.name: getattr(self, field_info.name)
            for field_info in fields(self)
        }

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "MidiData":
        """Create an instance of MidiData from a dictionary."""
        # Extract all fields of the current class
        cls_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data_dict.items() if k in cls_fields}
        return cls(**filtered_data)

    @classmethod
    def from_data(cls, __data: Union["MidiData", Dict[str, Any]]) -> "MidiData":
        """Create an instance of MidiData from another MidiData or a dictionary."""
        if isinstance(__data, cls):
            return cls.from_dict(__data.to_dict())
        elif isinstance(__data, dict):
            return cls.from_dict(__data)
        else:
            raise TypeError("Unsupported type for MidiData initialization")


class MidiTrans:
    r"""MIDI transcription and processing class.
    
    Args:
        file_list: 
            1) Can be a file path where each line is "wav_p|tg_p"
            2) Can be a list/tuple of files in format [[wav_p, tg_p], ...]
               For single file, can input as [wav_p, tg_p]
        sample_rate: Audio sample rate
        hop_length: Hop length for analysis
                NOTE: Can adjust hop length for different MIDI transcription precision
        f0_type: For clean vocals use 'pm', for noisy audio use 'rmvpe'
                pm processing speed:    ~17it/s
                rmvpe processing speed: ~20it/s
        
    Usage:
        >>> # 1. Using `run` method as generator
        >>> midi_trans = MidiTrans(file_list=[wav_p, tg_p])
        >>> for i in midi_trans.run()
        >>>     print(i)
        MidiData(file_name='0000', ...
        
        >>> # 2. Directly using __call__ method
        >>> midi_trans = MidiTrans()        # No file_list needed
        >>> midi_trans([wav_p, tg_p])
        MidiData(file_name='0000', ...
    """

    def __init__(
        self,
        file_list: Union[tuple, list, str] = None,
        *,
        sample_rate: int = 16000,
        hop_length: int = 320,
        f0_type: Literal["pm", "rmvpe"] = "pm",
    ) -> None:

        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.f0_type = f0_type
        self._lock = threading.Lock()  # For thread-safe operations

        self._read_file(file_list)

        if f0_type.lower() == "pm":  # for clean audio
            self.f0_predictor = F0_predictor.PMF0Predictor(
                hop_length, sampling_rate=self.sample_rate
            )
        elif f0_type.lower() == "rmvpe":  # for noisy audio
            self.f0_predictor = F0_predictor.RMVPEF0Predictor(
                hop_length, sampling_rate=self.sample_rate
            )
        else:
            raise NotImplementedError(f"not implement")

    @property
    def frame_rate(self) -> float:
        return self.hop_length / self.sample_rate

    def _read_file(self, file_list=None):
        if file_list is None:
            return
        self.wav_path = []
        self.textgrid_path = []
        if isinstance(file_list, str):
            # Input is a file containing [wav_path, textgrid_path, ...] pairs
            with open(file_list, "r") as f:
                for line in tqdm(f, desc="Loading files!"):
                    l = line.strip().split("|")
                    self.wav_path.append(l[0])
                    self.textgrid_path.append(l[1])
        else:
            _sample = file_list[0]
            # Input is tuple or list
            if isinstance(_sample, str) and len(file_list) == 2:
                # [wav_path, textgrid_path]
                self.wav_path, self.textgrid_path = [file_list[0]], [file_list[1]]
            elif isinstance(_sample, list) and len(_sample) == 2:
                for l in tqdm(file_list, total=len(file_list), desc="Loading files!"):
                    self.wav_path.append(l[0])
                    self.textgrid_path.append(l[1])
                # [[w_p, t_p], [w_p, t_p], ...]

    def run(self, max_workers=None):
        """Run processing with multithreading support.
        
        Args:
            max_workers: Maximum number of threads to use (None for auto)
            
        Yields:
            Processed MidiData objects
        """
        if self.f0_type == "rmvpe":
            assert max_workers == 1
            # TODO: add multi GPU on RMVPE
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for w, t in zip(self.wav_path, self.textgrid_path):
                futures.append(executor.submit(self.forward, [w, t]))
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    yield future.result()
                except Exception as e:
                    print(f"Error processing file: {e}")
            
    def __call__(self, path):
        return self.forward(path)

    def forward(self, path=None):
        """Main processing function."""
        wav_path, tg_path = path

        data = MidiData()  # Initialize data container

        # 1. Read TextGrid and extract information
        tg = TextGrid()
        tg.read(tg_path)

        (
            word_list,
            phone_list,
            word_duration,
            phone_duration,
            ph2word,  # [2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 1]
        ) = self._process_textgrid(tg, self.frame_rate)

        num_frames = sum(word_duration)  # Total frames
        assert sum(phone_duration) == sum(
            word_duration
        ), f"phoneme duration must be equal to word duration"

        # 2. Read audio
        info = torchaudio.info(wav_path)
        wav, sr = torchaudio.load(
            wav_path,
            frame_offset=0,
            num_frames=int(num_frames * self.frame_rate * info.sample_rate),
        )

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # Trim or pad audio to target length
        target_length = num_frames * self.hop_length  # Target audio length
        padding_needed = target_length - wav.shape[1]
        if padding_needed > 0:
            # Calculate padding amounts
            pad_start = 0
            pad_end = padding_needed

            wav = F.pad(wav, (pad_start, pad_end), "constant", 0)[0].numpy()
        else:
            wav = wav[0, :target_length].numpy()  # Trim if wav is longer than target

        ###################################
        # Extract F0 and convert to MIDI
        f0, uv = self.f0_predictor.compute_f0_uv(wav)
        f0, uv, note, delta_pitch, mid_hz, _ = get_midi(
            wav, self.sample_rate, f0, uv=uv
        )

        uv = np.array(uv, dtype=bool)
        if f0.shape[0] < num_frames:
            pad = math.ceil(num_frames) - f0.shape[0]
            f0 = np.pad(f0, [0, pad], mode="constant", constant_values=[0, f0[-1]])
            uv = np.pad(uv, [0, pad], mode="constant", constant_values=[0, uv[-1]])
            note = np.pad(
                note, [0, pad], mode="constant", constant_values=[0, note[-1]]
            )

        onset = np.diff(note, prepend=np.array([note[0]]))
        onset = np.where(onset != 0, 1, 0)
        note = self._nearest_neighbor_interpolation(phone_list, phone_duration, note)
        start_idx = 0  # Initialize start_idx
        ph_dur_start_idx = 0
        # word level / phoneme level
        midi, midi_ph = [], []
        midi2word, midi2ph = [], []
        midi_dur, midi_ph_dur = [], []

        for i, durations in enumerate(word_duration):
            end_idx = start_idx + durations
            ph_dur_end_idx = ph_dur_start_idx + ph2word[i]
            word_pitch = note[start_idx:end_idx]

            if word_list[i].strip().upper() in ["SP", "AP", "<SP>", "<AP>"]:
                # Silence handling: pitch=0 for entire segment
                pitches = [0] * durations

                # word level midi
                midi.extend([0])
                midi2word.append(1)
                midi_dur.extend([durations])

                # phoneme level midi
                midi_ph.extend([0])
                midi2ph.append(1)
                midi_ph_dur.extend([durations])

            else:
                pitches, durations = self._process_word_pitches(word_pitch)
                
                # update word level midi
                pitches, durations, ph_dur_i = self._adjust_note_boundaries(pitches,durations,phone_duration,
                                                                      ph_dur_start_idx,ph_dur_end_idx)
                midi.extend(pitches)
                midi2word.append(len(pitches))
                midi_dur.extend(durations)

                # Split word-level notes into phoneme-level notes based on phoneme durations
                ph_dur_i_cal = np.cumsum([0] + ph_dur_i)
                ph_frame_midi = np.repeat(pitches, durations)

                for ij in range(len(ph_dur_i_cal) - 1):
                    _ph_frame_midi, _ph_frame_midi_dur = self._process_word_pitches(
                        ph_frame_midi[ph_dur_i_cal[ij] : ph_dur_i_cal[ij + 1]]
                    )
                    midi_ph.extend(_ph_frame_midi)
                    midi_ph_dur.extend(_ph_frame_midi_dur)
                    midi2ph.append(len(_ph_frame_midi))

            start_idx = end_idx  # Update start_idx
            ph_dur_start_idx = ph_dur_end_idx

        assert len(midi) == sum(midi2word) and len(midi_ph) == sum(midi2ph)
        assert sum(midi_dur) == sum(midi_ph_dur)

        data.f0 = f0 * (~uv)

        data.midi = midi
        data.midi_dur = midi_dur
        data.midi_len = len(midi_dur)
        data.midi2word = midi2word

        data.midi_ph = midi_ph
        data.midi_ph_dur = midi_ph_dur
        data.midi_ph_len = len(midi_ph_dur)
        data.midi2ph = midi2ph

        data.midi_frame = np.repeat(midi, midi_dur)

        data.word = word_list
        data.word_dur = word_duration

        data.ph2word = ph2word
        data.ph = phone_list
        data.ph_dur = phone_duration
        data.audio = wav
        data.file_name = Path(wav_path).stem

        return data
    
    def _adjust_note_boundaries(self, pitches, durations, phone_duration, ph_start_idx, ph_end_idx):
        """
        Adjust note boundaries to match phoneme durations and handle fragmented notes.
        
        Args:
            pitches: List of MIDI pitch values
            durations: List of note durations (in seconds)
            phone_duration: List of phoneme durations
            ph_start_idx: Start index of current phoneme group
            ph_end_idx: End index of current phoneme group
        
        Returns:
            Modified pitches and durations lists
        """
        # Extract durations for current phoneme group
        ph_dur_i = phone_duration[ph_start_idx:ph_end_idx]
        first_ph_dur = ph_dur_i[0]  # Duration of first phoneme (typically consonant)
        first_note_dur = durations[0]  # Duration of first note

        # Handle multi-phoneme case (consonant+vowel) where consonant needs longer duration
        if len(ph_dur_i) > 1 and first_ph_dur > first_note_dur:
            j = 0
            while j < len(durations) - 1:
                # If current note is longer than next, merge next into current
                if durations[j] >= durations[j + 1]:
                    durations[j] += durations[j + 1]
                    pitches.pop(j + 1)
                    durations.pop(j + 1)
                else:
                    # Otherwise merge current note into next
                    durations[j + 1] += durations[j]
                    pitches.pop(j)
                    durations.pop(j)
                
                # Check if duration requirement is met
                if durations[j] >= first_ph_dur:
                    break
                
                # If we've reached last note, directly adjust its duration
                if j == len(durations) - 1:
                    durations[j] = first_ph_dur
                    break
        
        # Handle single-phoneme case (vowel only) with potential pitch fluctuations/ornaments
        min_note_duration = 0.1 / self.frame_rate  # Merge notes shorter than 100ms
        i = 0
        while i < len(durations) - 1:
            if durations[i] < min_note_duration:
                # Merge short note into next note
                durations[i + 1] += durations[i]
                pitches.pop(i)
                durations.pop(i)
            else:
                i += 1
        
        return pitches, durations, ph_dur_i
    
    @staticmethod
    def _nearest_neighbor_interpolation(phone_list, phone_duration, arr):
        """Interpolate MIDI notes with nearest neighbor approach."""
        frame_phone_list = []
        for p, ph_d in zip(phone_list, phone_duration):
            if p.upper() in ["SP", "AP", "<SP>", "<AP>"]:
                frame_phone_list.extend([0] * ph_d)
            else:
                frame_phone_list.extend([1] * ph_d)
        frame_phone_list = np.array(frame_phone_list)

        # Find indices and values of non-zero elements
        non_zero_indices = np.where(arr != 0)[0]
        non_zero_values = arr[non_zero_indices]

        # If no zeros in array, return as-is
        if len(non_zero_indices) == len(arr):
            return arr

        # Create interpolation function with 'next' method
        interp_func = interp1d(
            non_zero_indices, non_zero_values, kind="next", fill_value="extrapolate"
        )

        # Generate interpolated array
        interpolated_arr = interp_func(np.arange(len(arr)))
        for i in range(len(interpolated_arr) - 2, -1, -1):
            if np.isnan(interpolated_arr[i + 1]) and not np.isnan(interpolated_arr[i]):
                interpolated_arr[i + 1] = interpolated_arr[i]

        # Fill any remaining NaNs with 0
        interpolated_arr = np.nan_to_num(interpolated_arr, nan=0.0)
        interpolated_arr = np.where(
            frame_phone_list == 0, arr, interpolated_arr
        ).astype(np.int64)

        return interpolated_arr

    @staticmethod
    def _process_word_pitches(word_pitch):
        """Process pitch sequence for a word, merging single-frame notes."""
        if len(word_pitch) == 0:
            return [], []

        # Step 1: Identify all continuous pitch segments
        segments = []
        current_pitch = word_pitch[0]
        start_idx = 0

        for i in range(1, len(word_pitch)):
            if word_pitch[i] != current_pitch:
                segments.append((current_pitch, i - start_idx))
                current_pitch = word_pitch[i]
                start_idx = i
        segments.append((current_pitch, len(word_pitch) - start_idx))

        # Step 2: Merge single-frame segments
        merged_segments = []
        i = 0
        while i < len(segments):
            pitch, duration = segments[i]

            if duration == 1:  # Need to handle single-frame notes
                # Try merging with previous segment if exists and not single-frame
                if i > 0 and segments[i - 1][1] > 1:
                    merged_segments[-1] = (
                        merged_segments[-1][0],
                        merged_segments[-1][1] + 1,
                    )
                    i += 1
                # Otherwise try merging with next segment
                elif i < len(segments) - 1:
                    merged_segments.append((segments[i + 1][0], segments[i + 1][1] + 1))
                    i += 2
                # If no segments to merge with, keep as-is (though not ideal)
                else:
                    merged_segments.append((pitch, duration))
                    i += 1
            else:
                merged_segments.append((pitch, duration))
                i += 1

        # Step 3: Check again for single-frame notes (may have been created during merging)
        final_segments = []
        for pitch, duration in merged_segments:
            if duration == 1 and len(final_segments) > 0:
                final_segments[-1] = (final_segments[-1][0], final_segments[-1][1] + 1)
            else:
                final_segments.append((pitch, duration))

        # Convert to output format
        if not final_segments:
            return [], []

        pitches = [p for p, d in final_segments]
        durations = [d for p, d in final_segments]
        return pitches, durations

    @staticmethod
    def _process_textgrid(tg, timestep, epsilon=1e-6):
        """Process TextGrid file to extract word and phoneme information."""
        # Assume tg[0] is word tier, tg[1] is phone tier
        words = tg[0]
        phones = tg[1]

        # Initialize lists
        word_list = []
        phone_list = []
        word_duration = []
        phone_duration = []
        word2ph = []

        phone_index = 0  # Phone tier index

        for word in words:
            word_text = word.mark.strip()
            word_start = word.minTime
            word_end = word.maxTime

            # Skip empty words
            if word_text == "":
                continue

            # Calculate word duration and frames
            word_dur = word_end - word_start
            if word_dur < 0:
                print(f"Warning: Word '{word_text}' has negative duration. Skipping.")
                continue  # Skip words with negative duration

            word_frames = int(round(word_dur / timestep))
            word_list.append(word_text)

            num_phones_in_word = 0  # Number of phones in current word
            phones_in_word = []  # Phone info for current word

            # Process phones belonging to current word
            while phone_index < len(phones):
                phone = phones[phone_index]
                phone_text = phone.mark.strip()
                phone_start = phone.minTime
                phone_end = phone.maxTime

                # Skip empty phones and silences
                if phone_text in ["", "sp", "spn", "sil"]:
                    phone_index += 1
                    continue

                # Ensure positive phone duration
                if phone_end <= phone_start:
                    print(
                        f"Warning: Phone '{phone_text}' has invalid duration. Skipping."
                    )
                    phone_index += 1
                    continue

                # If phone ends before word starts, move to next phone
                if phone_end <= word_start - epsilon:
                    phone_index += 1
                    continue

                # If phone starts after word ends, finish processing current word
                if phone_start >= word_end + epsilon:
                    break

                # Determine if phone belongs to current word
                phone_mid = (phone_start + phone_end) / 2
                if word_start - epsilon <= phone_mid <= word_end + epsilon:
                    # Phone belongs to current word
                    phones_in_word.append(
                        {
                            "text": phone_text,
                            "start": phone_start,
                            "end": phone_end,
                            "duration": phone_end - phone_start,  # Duration is positive
                        }
                    )

                    num_phones_in_word += 1
                    phone_index += 1  # Move to next phone
                elif phone_mid < word_start - epsilon:
                    # Phone is before current word, move to next phone
                    phone_index += 1
                    continue
                elif phone_mid > word_end + epsilon:
                    # Phone is after current word, finish processing
                    break
                else:
                    # Other cases, move to next phone
                    phone_index += 1

            # Skip word if it contains no phones
            if num_phones_in_word == 0:
                continue

            # Calculate phone frames, ensuring total matches word frames
            total_phone_duration = sum([p["duration"] for p in phones_in_word])
            phone_frames_list = []

            # Distribute frames proportionally
            accumulated_frames = 0
            for idx, p in enumerate(phones_in_word):
                # Distribute frames based on duration proportion
                proportion = p["duration"] / total_phone_duration
                phone_frames = int(round(proportion * word_frames))
                accumulated_frames += phone_frames

                if idx == len(phones_in_word) - 1:
                    if word_frames > accumulated_frames:
                        # Last phone, ensure total frames match
                        phone_frames = word_frames - accumulated_frames
                    else:
                        phone_frames = accumulated_frames - word_frames
                        word_frames += accumulated_frames - word_frames
                phone_frames_list.append(phone_frames)

            # Adjust for rounding errors
            total_assigned_frames = sum(phone_frames_list)
            frame_difference = word_frames - total_assigned_frames
            if frame_difference != 0:
                phone_frames_list[-1] += frame_difference  # Adjust last phone's frames

            # Update lists
            word_duration.append(word_frames)
            word2ph.append(num_phones_in_word)

            for idx, p in enumerate(phones_in_word):
                if p["text"].upper() in ["SP", "<SP>", "AP", "<AP>"]:
                    phone_list.append(p["text"])
                else:
                    phone_list.append(p["text"])
                phone_duration.append(phone_frames_list[idx])

        word2ph_ = [idx + 1 for idx, count in enumerate(word2ph) for _ in range(count)]

        # Validate word2ph total equals phone_list length
        assert sum(word2ph) == len(phone_list), "word2ph total should equal phone_list length"
        assert len(word2ph_) == len(phone_list), "word2ph_ length should equal phone_list length"

        # Validate word_duration total equals phone_duration total
        assert sum(word_duration) == sum(
            phone_duration
        ), "word_duration total should equal phone_duration total"

        return word_list, phone_list, word_duration, phone_duration, word2ph

    def plot(self, data: MidiData, save_path="out.png"):
        """Plot MIDI and F0 over mel spectrogram."""
        mel = librosa.feature.melspectrogram(
            y=data.audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)  # Convert to dB

        fig, ax1 = plt.subplots(figsize=(12, 6))
        librosa.display.specshow(
            mel_db,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis="time",
            y_axis="mel",
            cmap="magma",
            vmin=-80,
            vmax=0,
            ax=ax1,  # Bind to left y-axis
        )

        ax2 = ax1.twinx()

        ax2.plot(
            librosa.frames_to_time(
                range(len(data.midi_frame)),
                sr=self.sample_rate,
                hop_length=self.hop_length,
            ),
            librosa.midi_to_hz(data.midi_frame),
            color="cyan",
            linewidth=2,
            alpha=0.8,
            label="MIDI",
        )

        ax2.plot(
            librosa.frames_to_time(
                range(len(data.f0)), sr=self.sample_rate, hop_length=self.hop_length
            ),
            data.f0,
            color="lime",
            linewidth=2,
            alpha=0.8,
            label="F0",
        )

        # Add vertical lines for word boundaries
        cumulative_dur = np.cumsum(data.word_dur)  # Cumulative frames
        # Convert to time points (seconds)
        time_points = librosa.frames_to_time(
            cumulative_dur, sr=self.sample_rate, hop_length=self.hop_length
        )
        # Draw dashed lines
        for t in time_points[:-1]:  # Skip last point (audio end)
            ax1.axvline(x=t, color="white", linestyle="--", linewidth=1, alpha=0.5)
            ax2.axvline(x=t, color="white", linestyle="--", linewidth=1, alpha=0.5)

        # Set right y-axis range (0-1200Hz)
        ax2.set_ylim(0, 600)
        ax2.set_ylabel("Frequency (Hz)")  # Right y-axis label
        ax2.tick_params(axis="y")  # Tick parameters

        # Set title and legend
        ax1.set_title("Mel Spectrogram with MIDI and F0")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.close()

    @staticmethod
    def save_midi(midi, midi_dur, frame_rate=0.01, midi_path="out.mid"):
        """
        Save MIDI file with silence handling.
        
        Args:
            midi: MIDI pitch array (0 indicates silence)
            midi_dur: Note duration array in frames
            frame_rate: Seconds per frame
            midi_path: Output path
        """
        notes = np.array(midi)
        durations = np.array(midi_dur) * frame_rate

        # Calculate time points
        start_times = np.cumsum(np.concatenate(([0], durations[:-1])))
        end_times = start_times + durations

        # Combine into interval array
        note_itv = np.column_stack((start_times, end_times))

        piano_chord = pretty_midi.PrettyMIDI()
        piano_program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
        piano = pretty_midi.Instrument(program=piano_program)

        for idx in range(len(notes)):
            pitch, itv = notes[idx], note_itv[idx]

            # Silence handling (pitch=0)
            if pitch == 0:
                # Add zero-velocity note
                note = pretty_midi.Note(
                    velocity=0,  # Zero velocity indicates silence
                    pitch=60,  # Arbitrary reasonable pitch (typically C4=60)
                    start=itv[0],
                    end=itv[1],
                )
            else:
                note = pretty_midi.Note(
                    velocity=120, pitch=int(pitch), start=itv[0], end=itv[1]
                )
            piano.notes.append(note)

        piano_chord.instruments.append(piano)
        piano_chord.write(str(midi_path))
        return piano_chord

if __name__ == "__main__":
    # Usage example 1
    # data = [
    #     "demo/0000.wav",
    #     "demo/0000.TextGrid",
    # ]
    # midi_trans = MidiTrans(
    #     file_list=None, sample_rate=44100, hop_length=441, f0_type="pm"
    # )
    
    # out = midi_trans(data)
    # midi_trans.plot(out)
    # print(out)

    # midi_trans.save_midi(out.midi, out.midi_dur, frame_rate=midi_trans.frame_rate)

    # Usage example 2 with multithreading
    data = [ [
        "demo/0000.wav",
        "demo/0000.TextGrid",
    ] for  _  in range(1000)]

    midi_trans = MidiTrans(file_list=data, sample_rate=44100, hop_length=441, f0_type='pm')
    
    # Process with multithreading
    for i in tqdm(midi_trans.run(max_workers=1)):
        pass
