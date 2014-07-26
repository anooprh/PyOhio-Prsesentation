import wave
import sys
import numpy
import scipy.fftpack
import struct
import math
from sklearn import preprocessing

if len(sys.argv) < 2:
    print("Extracts MFCCs from a wave file.\nUsage: %s filename.wav\n" % sys.argv[0])
    sys.exit(-1)


def get_frames(speech_sig, frame_length, frame_shift):

    remainder = speech_sig.shape[0] % frame_length
    speech_sig_zpad = numpy.append(speech_sig, numpy.zeros( (frame_length-remainder), ) )
    
    speech_frames =[]
    start_pos = 0
    while True:
        windowed_frame = speech_sig_zpad[start_pos:start_pos+frame_length]* numpy.hamming(frame_length)
        speech_frames.append(windowed_frame)
        start_pos = start_pos + frame_shift
        if start_pos+frame_length >= speech_sig_zpad.shape[0]:
            break

    return numpy.array(speech_frames)



def gen_mel_filts(num_filts, framelength, samp_freq):

    mel_filts = numpy.zeros((framelength, num_filts))
    step_size = int(framelength/float((num_filts + 1))) #Sketch it out to understand
    filt_width = math.floor(step_size*2)
    
    filt = numpy.bartlett(filt_width)
    
    step = 0
    for i in xrange(num_filts):
        mel_filts[step:step+filt_width, i] = filt
        step = step + step_size

    # Let's find the linear filters that correspond to the mel filters
    # The freq axis goes from 0 to samp_freq/2, so...
    samp_freq = samp_freq/2 

    filts = numpy.zeros((framelength, num_filts))
    for i in xrange(num_filts):
        for j in xrange(framelength):
            freq = (j/float(framelength)) * samp_freq

            # See which freq pt corresponds on the mel axis
            mel_freq = 1127 * numpy.log( 1 + freq/700  )
            mel_samp_freq = 1127 * numpy.log( 1 + samp_freq/700  )

            # where does that index in the discrete frequency axis
            mel_freq_index = int((mel_freq/mel_samp_freq) * framelength)
            if mel_freq_index >= framelength-1:
                mel_freq_index = framelength-1
            filts[j,i] = mel_filts[mel_freq_index,i]

    # Let's normalize each filter based on its width
    for i in xrange(num_filts):
        nonzero_els = numpy.nonzero(filts[:,i])
        width = len(nonzero_els[0])
        filts[:,i] = filts[:,i]*(10.0/width)

    return filts




wf = wave.open(sys.argv[1], 'rb')

samp_rate = wf.getframerate()

if wf.getnchannels() != 1:
    print "Oops. This code does not work with multichannel recordings. Please provide a mono file\n"
    sys.exit(-1)

frames = wf.readframes(wf.getnframes())
data = numpy.array(struct.unpack_from("%dh" % wf.getnframes(), frames))
wf.close()



# NOTE: These frames are not the same 'frames' as the ones in the wav file 
frame_size = 0.025 #in seconds
frame_shift = 0.0125 #in seconds
frame_length = int(samp_rate * frame_size)
frame_shift_length = int(samp_rate * frame_shift)
speech_frames = get_frames(data, frame_length, frame_shift_length)



#Let's compute the spectrum
comp_spec = numpy.fft.rfft(speech_frames,n=1024)
mag_spec = abs(comp_spec)
#numpy.savetxt('mag_spec.data',mag_spec)

# Mel warping
filts = gen_mel_filts(40, 513, samp_rate) # 1024 point FFT
mel_spec = numpy.dot(mag_spec,filts)
#numpy.savetxt('mel_spec.data', mel_spec)

# Mel log spectrum
mel_log_spec = mel_spec #trust me on this
nonzero = mel_log_spec > 0
mel_log_spec[nonzero] = numpy.log(mel_log_spec[nonzero])
#numpy.savetxt('mel_log_spec.data', mel_log_spec)

# Mel cepstrum
# mel_comp_cep = numpy.fft.rfft(mel_log_spec, n=76) #Not really complex cep
# mel_cep = mel_comp_cep.real
# # Discarding higher order cepstra
# numpy.savetxt('recomp_cep.data', mel_cep)
# mel_cep = mel_cep[:,0:13]

# Mel Cepstrum
mel_cep = scipy.fftpack.dct(mel_log_spec)
#numpy.savetxt('mel_cep.data', mel_cep)

#mel_recons_spec = scipy.fftpack.idct(mel_cep)
#numpy.savetxt('recomp.data', mel_recons_spec)

mel_cep = mel_cep[:,0:13]

#Mel Cep deltas
mel_cep_shift = numpy.delete(mel_cep,[0,1],axis=0)
blanks = numpy.zeros((2,mel_cep_shift.shape[1]))
mel_cep_shift = numpy.append(mel_cep_shift, blanks, axis=0)
mel_cep_deltas = mel_cep_shift - mel_cep
all_feats = numpy.append(mel_cep,mel_cep_deltas, axis=1)

#Mel Cep Delta-deltas
mel_cep_shift = numpy.delete(mel_cep_deltas,[0,1],axis=0)
mel_cep_shift = numpy.append(mel_cep_shift, blanks, axis=0)
mel_cep_delta_deltas = mel_cep_shift - mel_cep_deltas
all_feats = numpy.append(all_feats, mel_cep_delta_deltas, axis=1)

# Cepstral Mean and Variance Normalization                                         
all_feats_norm = preprocessing.scale(all_feats)


numpy.savetxt('mel_cep.data', all_feats_norm)
