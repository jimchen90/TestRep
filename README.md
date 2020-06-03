# Bass_biquad_test
This file is to test the bass biquad filter and compare coefficients and output waveforms with SoX and MATLAB audio toolbox.
The result summary is shown in the beginning of notebook. More details can be seen in https://github.com/pytorch/audio/issues/676 and https://github.com/pytorch/audio/pull/661




# bass_biquad
This file is to implement a sox dependency reduction task for bass (with biquad) following (https://github.com/pytorch/audio/issues/260). 
Its method can be based on the implementation of task for treble with biquad from 
(https://github.com/pytorch/audio/blob/master/torchaudio/functional.py#L1025).
Here we want to compare and test the result from sox and our bass_biquad function.
