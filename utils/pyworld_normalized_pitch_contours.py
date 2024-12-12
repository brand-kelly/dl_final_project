from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pickle
import numpy as np
from scipy.interpolate import interp1d
import pywt
import h5py


def interpolate_f0(f0):
    voiced_indices = np.where(f0 > 0)[0]
    if len(voiced_indices) == 0:
        return f0
        
    interp_function = interp1d(
        voiced_indices, f0[voiced_indices], kind="linear", fill_value="extrapolate"
    )
    
    interpolated_f0 = interp_function(np.arange(len(f0)))

    return interpolated_f0

def convert_to_log_f0(f0):
    safe_f0 = np.where(f0 > 0, f0, 1e-5)
    log_f0 = np.log(safe_f0)
    return log_f0


def normalize_log_f0(log_f0):
    mean = np.mean(log_f0)
    std = np.std(log_f0)
    normalized_log_f0 = (log_f0 - mean) / (std + 1e-6)
    return normalized_log_f0

def process_spectrogram(pitch_contour):
    try:
        tmp_dict = {}

        f0 = interpolate_f0(pitch_contour["f0"])
        log_f0 = convert_to_log_f0(f0)
        norm_log_f0 = normalize_log_f0(log_f0)
        tmp_dict["id"] = pitch_contour["id"]
        tmp_dict["norm_log_f0"] = norm_log_f0
        tmp_dict["time_axis"] = pitch_contour["time_axis"]
        return tmp_dict

    except Exception as e:
        print(f"Error processing {pitch_contour['id']}: {e}")
        return None


def process_all_pitch_contours(pyworld_pitch_contours, num_workers=4):
    pyworld_pitch_spectrograms = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_spectrogram, pyworld_pitch_contours),
                total=len(pyworld_pitch_contours),
            )
        )

    pyworld_pitch_spectrograms.extend(
        [r for r in results if r is not None]
    )
    
    return pyworld_pitch_spectrograms


def main():
    
    with open("./data/pyworld_pitch_contours-3500.pkl", "rb") as f:
        pyworld_pitch_contours = pickle.load(f)
        
    num_workers = 10
    print(f"Using {num_workers} workers for parallel processing.")

    # Process pitch contours
    pyworld_norm_pitch_contours = process_all_pitch_contours(
        pyworld_pitch_contours, num_workers
    )

    with open("./data/pyworld_norm_pitch_contours-3500.pkl", "wb") as f:
        pickle.dump(pyworld_norm_pitch_contours, f)

    print(f"Processed {len(pyworld_norm_pitch_contours)} files.")

if __name__ == "__main__":
    main()
