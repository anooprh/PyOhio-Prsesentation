import pyaudio
import wave
import array
import numpy

CHUNK = 1225
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 40
WAVE_OUTPUT_FILENAME = "output.wav"

threshold = 200

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

num_sil_frames = 0

# The device will record for 20 seconds until there has been 0.5
# second of audio (18 frames) where the energy is less than the
# threshold
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
    nums = array.array('h',data)
    left = numpy.array(nums[1::2])
    energy = 10*numpy.log(numpy.sum(numpy.power(left,2)))
    #print(energy)

    if (energy < threshold):
        num_sil_frames = num_sil_frames + 1
    else:
        num_sil_frames = 0

    if num_sil_frames >= 37:
        break

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()



wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
