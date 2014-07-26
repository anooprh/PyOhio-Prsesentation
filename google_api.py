import urllib2
import sys
import json

if len(sys.argv) < 2:
    print("Extracts text from a single file using google Speech API.\nUsage: %s <audio_filename>" % sys.argv[0])
    sys.exit(-1)


url = "https://www.google.com/speech-api/v2/recognize?output=json&lang=en-us&key=AIzaSyA0iog2ipq2wrPyuAB7hWNMphvkR_yMgbM&results=6&pfilter=2"
audio = open(sys.argv[1]).read()
headers={'Content-Type': 'audio/x-flac; rate=44100'}
request = urllib2.Request(url, data=audio, headers=headers)
response = urllib2.urlopen(request)

print response.read()