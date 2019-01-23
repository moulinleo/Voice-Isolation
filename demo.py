from demo_util import import_model, prepare_inputdata, prepare_outputdata
import librosa
import winsound
from winsound import *

'''
Demo of Voice Separation
'''

# Choice of song & duration (in sec)
namesong = 'LizNelson_Rainfall_snipped.wav'
duration = 30.0
threshold = 0.7

# Load song
mix, sr = librosa.load(path=namesong, duration=duration)

# Import pre-trained model
model = import_model()

# Prepare input data
spectro_mix, phase = prepare_inputdata(mix)

# Voice separation
mask_pred = model.predict(spectro_mix)

# Prepare output data
result_voice, result_instru = prepare_outputdata(mask_pred, spectro_mix, phase, threshold)

# Save result
librosa.output.write_wav('result.wav', result_voice, sr)

# Play result
print("Playing result song..")
winsound.PlaySound('result.wav', SND_FILENAME)



