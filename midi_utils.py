
import librosa
import numpy as np
from librosa.feature.rhythm import tempo
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.signal import medfilt

def one_beat_frame_size(tempo):
    """ Calculate frame size of 1 beat
    ----------
    Parameters:
        tempo: float
    
    ----------
    Returns: 
        tempo: int
    
    """
    return int(np.round(60 / tempo * 100))

def median_filter_pitch(pitch, medfilt_size, weight):
    """ Smoothing pitch using median filter
    ----------
    Parameters:
        pitch: array
        medfilt_size: int
        weight: float
    
    ----------
    Returns: 
        pitch: array
    
    """

    medfilt_size = int(medfilt_size * weight)
    if medfilt_size % 2 == 0:
        medfilt_size += 1
    return np.round(medfilt(pitch, medfilt_size))


def clean_note_frames(note, min_note_len=5):
    """ Remove short pitch frames 
    ----------
    Parameters:
        note: array
        min_note_len: int
        
    ----------
    Returns: 
        output: array
    
    """

    prev_pitch = 0
    prev_pitch_start = 0
    output = np.copy(note)
    for i in range(len(note)):
        pitch = note[i]
        if pitch != prev_pitch:
            prev_pitch_duration = i - prev_pitch_start
            if prev_pitch_duration < min_note_len:
                output[prev_pitch_start:i] = [0] * prev_pitch_duration
            prev_pitch = pitch
            prev_pitch_start = i
    return output

def makeSegments(note):
    """ Make segments of notes
    ----------
    Parameters:
        note: array
               
    ----------
    Returns: 
        startSeg: starting points (array)
        endSeg: ending points (array)
    
    """
    startSeg = []
    endSeg = []
    flag = -1
    if note[0] > 0:
        startSeg.append(0)
        flag *= -1
    for i in range(0, len(note) - 1):
        if note[i] != note[i + 1]:
            if flag < 0:
                startSeg.append(i + 1)
                flag *= -1
            else:
                if note[i + 1] == 0:
                    endSeg.append(i)
                    flag *= -1
                else:
                    endSeg.append(i)
                    startSeg.append(i + 1)
    return startSeg, endSeg

def remove_short_segment(idx, note_cleaned, start, end, minLength):
    """ Remove short segments
    ----------
    Parameters:
        idx: (int)
        note_cleaned: (array)
        start: starting points (array)
        end: ending points (array)
        minLength: (int)
               
    ----------
    Returns: 
        note_cleaned: (array)
            
    """

    len_seg = end[idx] - start[idx]
    if len_seg < minLength:
        if (start[idx + 1] - end[idx] > minLength) and (start[idx] - end[idx - 1] > minLength):
            note_cleaned[start[idx] : end[idx] + 1] = [0] * (len_seg + 1)
    return note_cleaned


def remove_octave_error(idx, note_cleaned, start, end):
    """ Remove octave error
    ----------
    Parameters:
        idx: (int)
        note_cleaned: (array)
        start: starting points (array)
        end: ending points (array)
               
    ----------
    Returns: 
        note_cleaned: (array)
            
    """
    len_seg = end[idx] - start[idx]
    if (note_cleaned[start[idx - 1]] == note_cleaned[start[idx + 1]]) and (
        note_cleaned[start[idx]] != note_cleaned[start[idx + 1]]
    ):
        if np.abs(note_cleaned[start[idx]] - note_cleaned[start[idx + 1]]) % 12 == 0:
            note_cleaned[start[idx] - 1 : end[idx] + 1] = [note_cleaned[start[idx + 1]]] * (
                len_seg + 2
            )
    return note_cleaned


def clean_segment(note, minLength):
    """ clean note segments
    ----------
    Parameters:
        note: (array)
        minLength: (int)
               
    ----------
    Returns: 
        note_cleaned: (array)
            
    """

    note_cleaned = np.copy(note)
    start, end = makeSegments(note_cleaned)

    for i in range(1, len(start) - 1):
        note_cleaned = remove_short_segment(i, note_cleaned, start, end, minLength)
        note_cleaned = remove_octave_error(i, note_cleaned, start, end)
    return note_cleaned

def refine_note(est_note, tempo):
    """ main: refine note segments
    ----------
    Parameters:
        est_note: (array)
        tempo: (float)
               
    ----------
    Returns: 
        est_pitch_mf3_v: (array)
            
    """
    one_beat_size = one_beat_frame_size(tempo)
    est_note_mf1 = median_filter_pitch(est_note, one_beat_size, 1 / 8)
    est_note_mf2 = median_filter_pitch(est_note_mf1, one_beat_size, 1 / 4)
    est_note_mf3 = median_filter_pitch(est_note_mf2, one_beat_size, 1 / 3)
    est_note_mf3 = np.nan_to_num(est_note_mf3,nan=0)
        
    vocing = est_note_mf1 > 0
    # print(vocing.shape)
    est_pitch_mf3_v = vocing * est_note_mf3
    est_pitch_mf3_v = clean_note_frames(est_pitch_mf3_v, int(one_beat_size * 1 / 8))
    est_pitch_mf3_v = clean_segment(est_pitch_mf3_v, int(one_beat_size * 1 / 4))
    return est_pitch_mf3_v


def interpolate_f0(f0, threshold=10):
    interpolated_indices = []
    interpolated_values = []
    zero_count = 0

    for i, value in enumerate(f0):
        if value == 0:
            zero_count += 1
            if zero_count == 1:
                start_index = i - 1
        else:
            if zero_count > 0:
                if zero_count < threshold:
                    end_index = i
                    indices = np.arange(start_index, end_index + 1)
                    values = f0[indices]
                    indices = indices[values != 0]
                    values = values[values != 0]
                    if len(indices) > 1:
                        f = interp1d(indices, values, kind='linear', fill_value='extrapolate', assume_sorted=True)
                        interpolated_indices.extend(range(start_index + 1, end_index))
                        interpolated_values.extend(f(range(start_index + 1, end_index)))
                    else:
                        interpolated_indices.append(start_index)
                        interpolated_values.append(values[0])
                zero_count = 0
            interpolated_indices.append(i)
            interpolated_values.append(value)

    interpolated_f0 = np.full_like(f0, 0., dtype=np.float64)
    interpolated_f0[interpolated_indices] = interpolated_values

    return interpolated_f0

def get_midi(audio, sr, f0, uv):

    f0 = interpolate_f0(f0, 5)
    uv = np.array(1 - uv, dtype=bool)
    f0 = np.where(uv, 1e-5, f0)
    midi = librosa.hz_to_midi(f0).round()
    midi *= (~uv)
    mid_midi = medfilt(midi, kernel_size=5)
    onset_strength = librosa.onset.onset_strength(y=audio, sr=sr)
    tempo_ = tempo(y=audio,sr=sr,onset_envelope=onset_strength)
    refined_fl_note = refine_note(mid_midi, tempo_) 
    note_rest = refined_fl_note < 10
    
    if not np.all(note_rest == False):
        try:
            interp_func = interpolate.interp1d(
                np.where(~note_rest)[0], refined_fl_note[~note_rest],
                kind='nearest', fill_value='extrapolate'
            )
            refined_fl_note[note_rest] = interp_func(np.where(note_rest)[0])
        except:
            return None,None,None,None,None,None
    
    mid_hz = librosa.midi_to_hz(refined_fl_note) * (~uv)

    if not np.all(uv == False):
        interp_func = interpolate.interp1d(
            np.where(~uv)[0], f0[~uv],
            kind='nearest', fill_value='extrapolate'
        )
        f0[uv] = interp_func(np.where(uv)[0])
        
    delta_pitch = np.clip(f0-mid_hz,a_min=-100,a_max=100)
    
    f0 = np.nan_to_num(f0 *(~uv),nan=0)
    mid_hz = np.nan_to_num(mid_hz*(~uv) ,nan=0)
    delta_pitch = np.nan_to_num(delta_pitch * (~uv), nan=0)
    mid_note =  refined_fl_note * (~uv)

    changes = np.where(np.diff(mid_note) != 0)[0] + 1
    segments = np.concatenate(([0], changes, [len(mid_note)]))
    mid_dur = np.diff(segments)
    return f0, uv, mid_note, delta_pitch, mid_hz, mid_dur
