# asr_pipeline_exercise
continuous streaming voice to text using open source libraries

As always use good virtualenv hygiene 

Things you'll need to pip install:

- pytorch (ideally get cuda version if you have an nvidia card)
- transformers[torch]
- soundfile
- librosa
- sounddevice


Tips:
sounddevice defaults to your default communications devices to record/playback
To check what those defaults are, use 

python -m sounddevice

and change accodingly as you need
