import sys
import librosa
import sounddevice as sd
import numpy as np
from queue import Queue
import threading
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Set the sampling rate (number of samples per second)
sr = 16000

# Create an output array to store the recorded audio
out = np.zeros((0, 1))

# Create a processing queue
q = Queue()

# Define a callback function to handle incoming audio
def audio_callback(indata, outdata, frames, time, status):
    # print("indata: " + str(indata) + "shape: " + str(indata.shape))
    # print("indata shape: " + str(indata.shape))
    # outdata[:] = indata
    global out, q
    data = indata.T.reshape(-1,)
    # print("Min: " + str(data.min()) + "\tMax: " + str(data.max()))
    # print("pre-trim data shape: " + str(data.shape))
    data, _ = librosa.effects.trim(data, top_db=5)
    # print("post-trim data shape: " + str(data.shape))
    # print("Min: " + str(data.min()) + "\tMax: " + str(data.max()))
    # print("data: " + str(data) + "shape: " + str(data.shape))
    # print("data.T shape: " + str(data.shape))
    # print("frames: " + str(frames))
    # print("time: " + str(time))
    # print("status: " + str(status))
    

    if status:
        print(status, file=sys.stderr)
    # Convert the input data to mono and append it to the output array
    # data = librosa.to_mono(data) #.reshape(-1, 1)
    # print("data shape after to_mono: " + str(data.shape))
    # print(data)
    # data = data.reshape(-1, 1)
    # print("data shape after reshape(-1,1): ")
    # print(data.shape)
    # print(data)
    # print("data captured: " + str(data) + " shape: " + str(data.shape))
    if len(out) == 0:
        out = data
    else:
        # out = np.hstack([out, data])
        out = np.concatenate((out, data))
    # print("length of out: " + str(out.nbytes))
    # print("shape of out: " + str(out.shape))
    # print("out: ")
    # print(out)
    # If the output array is larger than 1MB, trim it and send it to the processing queue
    if out.nbytes >= 262144:
        # Trim the output array to remove leading and trailing silence
        # trimmed, _ = librosa.effects.trim(out, top_db=20)
        # If the trimmed output array is not empty, send it to the processing queue
        if len(out) > 0:
            # print("sending new chunk to q: " + str(out.shape) + " - " + str(out.nbytes))
            q.put(out)
            # print("q size: " + str(q.qsize()))

        # Create a new output array
        out = np.zeros((0, 1))
    
# Define a processing function to process the output arrays from the queue
def process_output(q):
    # Create a WhisperProcessor object
    processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en")
    print("Model loaded")
    while True:
        # Wait for an output array to be added to the queue
        sample = q.get()
        if sample is None:
            print("no sample received, stopping")
            # If the output array is None, back to top
            break
        # print("sample acquired, processing...")
        # print("Sample shape: " + str(sample.shape))
        # print("Sample: ")
        # print(sample)
        # Get the input features from the output array using the WhisperProcessor
        # print("generating input features")
        # sd.play(sample, sr)
        input_features = processor(sample, sampling_rate=16000, return_tensors="pt").input_features
        # print("generating predicted ids")
        predicted_ids = model.generate(input_features)
        # stop
        # print("transcribing predicted ids to text")
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        if str(transcription[0].strip())  != "you":
            print("**transcription result: " + str(transcription[0]))

# Start the processing thread
processing_thread = threading.Thread(target=process_output, args=(q,))
processing_thread.start()
print("processing thread started")
# Start recording audio using the callback function
with sd.Stream(channels=1, callback=audio_callback, blocksize=2048, samplerate=sr):
    print("starting input stream with devices: " + str(sd.default.device))
    # sd.sleep(int(5 * 1000))
    while True:
        # Wait for user input to stop recording
        input_data = input()
        if input_data.lower() == 'stop':
            break

# Trim any remaining samples in the output array to remove leading and trailing silence
trimmed, _ = librosa.effects.trim(out, top_db=20)
# If the trimmed output array is not empty, send it to the processing queue
if len(trimmed) > 0:
    q.put(trimmed)

# Signal the processing thread to stop
q.put(None)
processing_thread.join()

# # Save the recorded audio as a WAV file using librosa
# librosa.output.write_wav('recording.wav', out, sr)
