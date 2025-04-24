from scipy.io import wavfile
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np

# Play the audio (this works in Jupyter notebooks)
def play_audio(sample_data, volume=1.0, sample_rate=44100):
    # Normalize to 16-bit range for WAV file
    audio = np.int16(sample_data * volume * 32767)
    
    print(sample_rate)

    # Save as WAV file
    wavfile.write("sine_wave.wav", sample_rate, audio)
    
    # Display audio player in notebook
    return ipd.Audio(audio, rate=sample_rate)

# Plot the waveform
def plot_waveform(sample_data, sample_rate=44100):
    time = np.linspace(0, len(sample_data)/sample_rate, len(sample_data))
    plt.figure(figsize=(10, 4))
    plt.plot(time, sample_data)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Sine Wave (440 Hz)")
    plt.xlim(0, 0.01)  # Show only first 10ms to see the wave shape clearly
    plt.grid(True)
    plt.show()