import numpy as np
import os
from PitchEstimationAMF import PitchEstimation
from matplotlib import pyplot as plt
from utils import get_label
import pandas as pd
# Refernce https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.590.5684&rep=rep1&type=pdf
SAMPLE_DIR = r"D:\DUT\XuLiTinHieuSo\PitchEstimation\TinHieuKiemThu"
OUTPUT_DIR = r"D:\DUT\XuLiTinHieuSo\PitchEstimation\outputs"


if __name__ == '__main__':
    # Get labels
    labels = get_label(SAMPLE_DIR)
    # Pitch estimate
    detector = PitchEstimation(frame_time=0.02, overlap_frame=0.3)
    # All sample file
    wave_names = os.listdir(SAMPLE_DIR)
    wave_names = [item for item in wave_names if item.endswith(".wav")]
    # Size of figure
    plt.rcParams["figure.figsize"] = (10, 10)
    # Metrics data
    result = {
        "Wave name": [],
        "F0 mean predict": [],
        "F0 mean label": [],
        "F0 mean difference": [],
        "F0 std predict": [],
        "F0 std label": [],
        "F0 std difference": []
    }
    # All wave loop
    for index, wave_name in enumerate(wave_names):
        file_name = wave_name[:wave_name.rfind(".")]
        print(">> Processing on " + wave_name)
        wave_path = os.path.join(SAMPLE_DIR, wave_name)
        detector.read_audio(wave_path)
        pitch, unvoice_amdf, unvoice_frame, voice_amdf, voice_frame \
            = detector.amf(median_size=5, threshold=0.35, label=labels[file_name])
        fs, signal = detector.get_audio()
        # Plot result
        # create position pair
        sample_axis = [0]
        pitch_axis = [0]
        for i in range(len(pitch)):
            if pitch[i] != 0:
                sample_axis.append(i)
                pitch_axis.append(pitch[i])
        sample_axis.append(len(pitch))
        pitch_axis.append(0)
        plt.figure(index)
        # Plot the frequency
        plt.subplot(4, 1, 1).scatter(sample_axis, pitch_axis, 5)
        plt.xlabel("sample index")
        plt.ylabel("frequency (Hz)")
        plt.title("Pitch estimate")
        # Plot the signal
        plt.subplot(4, 1, 2).plot(signal)
        plt.xlabel("sample index")
        plt.ylabel("amplitude")
        plt.title(wave_name)
        # Plot the unvoice amdf
        plt.subplot(4, 1, 3).plot(unvoice_amdf)
        plt.xlabel("lag value")
        plt.ylabel("amdf value")
        plt.title("Unvoice amdf")
        plt.ylim(0, 1)
        # Plot the voice amdf
        plt.subplot(4, 1, 4).plot(voice_amdf)
        plt.xlabel("lag value")
        plt.ylabel("amdf value")
        plt.title("Voice amdf")
        plt.ylim(0, 1)
        # Mean and Std of fs
        fs_mean = np.mean(pitch_axis[1:-1])
        fs_std = np.std(pitch_axis[1:-1])
        fs_mean_label = labels[file_name]['mean']
        fs_std_label = labels[file_name]['std']
        # Log mean and std
        result["Wave name"].append(file_name)
        result["F0 mean predict"].append(fs_mean)
        result["F0 mean label"].append(fs_mean_label)
        result["F0 mean difference"].append(abs(fs_mean - fs_mean_label))
        result["F0 std predict"].append(fs_std)
        result["F0 std label"].append(fs_std_label)
        result["F0 std difference"].append(abs(fs_std - fs_std_label))
        # Save
        save_path = os.path.join(OUTPUT_DIR, file_name + ".png")
        plt.savefig(save_path)
        plt.tight_layout(h_pad=3)
    plt.show()
    # Show metrics
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(pd.DataFrame(result)[["Wave name", "F0 mean predict", "F0 mean label", "F0 mean difference"]])
    print(pd.DataFrame(result)[["Wave name", "F0 std predict", "F0 std label", "F0 std difference"]])

