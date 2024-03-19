import os
import azure.cognitiveservices.speech as speechsdk

SPEECH_KEY="58e51ddb823747a3b65e1c0553f94f4e"
SPEECH_REGION="germanywestcentral"

# This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
speech_config1 = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
speech_config2 = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)

# audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
audio_config1 = speechsdk.audio.AudioOutputConfig(filename="1.wav")
audio_config2 = speechsdk.audio.AudioOutputConfig(filename="2.wav")

# The language of the voice that speaks.
speech_config1.speech_synthesis_voice_name='en-US-AvaNeural'
speech_config2.speech_synthesis_voice_name='en-US-AndrewNeural'
# en-US-AndrewNeural
# en-US-AvaNeural

speech_synthesizer1 = speechsdk.SpeechSynthesizer(speech_config=speech_config1, audio_config=audio_config1)
speech_synthesizer2 = speechsdk.SpeechSynthesizer(speech_config=speech_config2, audio_config=audio_config2)

# Get text from the console and synthesize to the default speaker.
# print("Enter some text that you want to speak >")
# text = input()
text1 = "As illustrated in Fig.1, the backbone of FreeVC is inherited from VITS, which is a CVAE augmented with GAN training."
text2 = "Different from VITS, the prior encoder of FreeVC takes raw waveform as input instead of text annotation, and has a different structure."


speech_synthesis_result1 = speech_synthesizer1.speak_text_async(text1).get()
speech_synthesis_result2 = speech_synthesizer2.speak_text_async(text2).get()

if speech_synthesis_result1.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
    print("Speech synthesized for text [{}]".format(text1))
elif speech_synthesis_result1.reason == speechsdk.ResultReason.Canceled:
    cancellation_details = speech_synthesis_result1.cancellation_details
    print("Speech synthesis canceled: {}".format(cancellation_details.reason))
    if cancellation_details.reason == speechsdk.CancellationReason.Error:
        if cancellation_details.error_details:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")