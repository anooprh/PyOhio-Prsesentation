20 raw audio recording of cats and dogs are in cat/ and dog/ directory respectively

Format the raw audio recordings to formatted ones and extract mfcc features

    ./format_animals.sh dog
    
    ./format_animals.sh cat

Ignore RiFF Id errors it gives here. This does not affect the functionality
    
Next test the result of training data on one of the remaining input audio files

for example
    ./live_demo_animal.sh cat/sound18.wav