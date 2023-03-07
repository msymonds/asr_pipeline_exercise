import sys
import sounddevice as sd
import numpy as np
from queue import Queue
import threading
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Set the sampling rate (number of samples per second)
sr = 16000

# set a signal floor to identify "silence"
signal_floor = 0.003

# Create an output array to store the recorded audio
out = np.zeros((0, 1))

# Create a processing queue
q = Queue()

# let's track how many blocks of silence we hear
silence_count = 0


# Define a callback function to handle incoming audio
def audio_callback(indata, outdata, frames, time, status):
    global out, q, silence_count, signal_floor

    data = indata.T.reshape(-1,)
    if data.max() < signal_floor:
        silence_count += 1
    else:
        silence_count = 0    

    if status:
        print(status, file=sys.stderr)
    
    if len(out) == 0:
        out = data
    else:
        out = np.concatenate((out, data))
    
    # If there's a sufficient length of silence, 
    # or a cumulative chunk larger than 250KB,
    # send it to the processing queue
    if silence_count == 3 or out.nbytes >= 262144:
        # don't q chunks with all silence
        if out.max() < signal_floor:
            out = np.zeros((0, 1))
            return
        q.put(out)
        # Create a new output array
        out = np.zeros((0, 1))
    
# Define a processing function to process the output arrays from the queue
def process_output(q):
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
        # Get the input features from the output array using the WhisperProcessor
        # sd.play(sample, sr)
        input_features = processor(sample, sampling_rate=sr, return_tensors="pt").input_features
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        # somehow gibberish or random noise cause it to output "you" or "You"
        if str(transcription[0].strip().lower())  != "you":
            print("**transcription result: " + str(transcription[0]))
        

# Start the processing thread
processing_thread = threading.Thread(target=process_output, args=(q,))
processing_thread.start()
print("processing thread started")
# Start recording audio using the callback function
with sd.Stream(channels=1, callback=audio_callback, blocksize=2048, samplerate=sr):
    print("starting input stream with devices: " + str(sd.default.device))
    while True:
        # Wait for user input to stop recording
        input_data = input()
        if input_data.lower() == 'stop':
            break

# send the last chunk to the processing queue
q.put(out)

# Signal the processing thread to stop
q.put(None)
processing_thread.join()

