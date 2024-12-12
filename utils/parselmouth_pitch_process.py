from concurrent.futures import ProcessPoolExecutor
import parselmouth
from tqdm import tqdm
import os
import pickle

def extract_pitch_contour(audio_path):
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    f0 = pitch.selected_array['frequency']
    time_axis = pitch.xs()
    return f0, time_axis

def process_pitch(mel):
    try:
        tmp_dict = {}
        # Load waveform
        f0, time_axis = extract_pitch_contour(mel['wav_path'])

        tmp_dict["id"] = mel["id"]
        tmp_dict["f0"] = f0
        tmp_dict["time_axis"] = time_axis
        return tmp_dict
    except Exception as e:
        print(f"Error processing {mel['id']}: {e}")
        return None  # Return None for failed items

def process_all_pitch_contours(mel_data, num_workers=4):
    parselmouth_pitch_contours = []

    # Use ProcessPoolExecutor for parallel processing with progress bar
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_pitch, mel_data), total=len(mel_data)))

    parselmouth_pitch_contours.extend([r for r in results if r is not None])  # Filter out None
    return parselmouth_pitch_contours


def main():
     # Load input data
    with open("./data/mel_data.pkl", "rb") as f:
        mel_data = pickle.load(f)

    # Set number of workers dynamically
    num_workers = 8
    print(f"Using {num_workers} workers for parallel processing.")

    # Process pitch contours
    parselmouth_pitch_contours = process_all_pitch_contours(mel_data, num_workers)

    # Save results
    with open("./data/parselmouth_pitch_contours.pkl", "wb") as f:
        pickle.dump(parselmouth_pitch_contours, f)

    print(f"Processed {len(parselmouth_pitch_contours)} files.")


if __name__ == "__main__":
    main()
