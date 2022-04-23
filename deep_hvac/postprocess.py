"""Functions to help in processing the results."""

import numpy as np
import pandas as pd


def results_to_dataframes(results_dict):
    out = {}
    for k, v in results_dict.items():
        if k == 'timestamp':
            out[k] = v[0]
            continue
        out[k] = pd.DataFrame(
            np.array(v).T,
            index=results_dict['timestamp'][0]
        )
    return out


def to_multiindex(results_frames):
    """Concatenate results into a multiindex DataFrame."""
    frames = results_frames.copy()
    if 'timestamp' in frames:
        frames.pop('timestamp')
    return pd.concat(frames.values(), axis=1, keys=frames.keys())


def expert_results_to_frame(results_dict):
    timestamps = results_dict.pop('timestamp')
    dataframe_cols = {}
    for k, v in results_dict.items():
        if k == 'timestamps':
            continue
        if len(v) != len(timestamps):
            dataframe_cols[k] = v[1:-1]
        else:
            dataframe_cols[k] = v
    return pd.DataFrame(dataframe_cols, index=timestamps)
