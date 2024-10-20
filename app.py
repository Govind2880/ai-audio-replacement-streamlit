import streamlit as st
import moviepy.editor as mp
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech
import requests
import os
import re
import json
from collections import defaultdict

# Load Google Cloud credentials from Streamlit secrets
google_credentials = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]

# Convert the AttrDict to a regular dictionary
google_credentials_dict = dict(google_credentials)

# Write the Google Cloud credentials to a file
with open("google_credentials.json", "w") as f:
    f.write(json.dumps(google_credentials_dict))

# Set the environment variable for Google Cloud authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_credentials.json"

AZURE_OPENAI_KEY = st.secrets["azure"]["AZURE_OPENAI_KEY"]

AZURE_OPENAI_ENDPOINT = "https://internshala.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"

# Set a limit for the maximum audio length (in seconds)
AUDIO_LIMIT_SECONDS = 59

st.title("AI Generated Audio Replacement in Video")

# Gender selection for the speaker
gender = st.selectbox("Select the speaker's gender:", ("Male", "Female"))

voice_name = "en-US-Journey-F" if gender == "Female" else "en-US-Journey-D"

# Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# Initialize a flag for proceeding
proceed = False

if uploaded_file is not None:
    # Save the uploaded file to disk
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("Uploaded video successfully saved.")

    # Extract audio from the uploaded video
    try:
        video = mp.VideoFileClip("uploaded_video.mp4")
        audio_duration = video.audio.duration

        st.write(f"Extracted video with duration: {audio_duration} seconds.")

        # Check if the audio exceeds the allowed limit
        if audio_duration > AUDIO_LIMIT_SECONDS:
            st.error(f"The audio duration exceeds the {AUDIO_LIMIT_SECONDS+1} second limit.")
            
            # Ask the user if they want to trim the video/audio
            trim_option = st.radio(
                "Your video is too long. Would you like to trim it to fit the required limit?",
                ("Yes, trim the video to 60 seconds", "No, I want to upload a shorter video")
            )

            # Show the "Proceed" button after selecting the trim option
            if st.button("Proceed"):
                proceed = True
        else:
            st.success("The video is within the allowed duration limit.")
            if st.button("Proceed"):
                proceed = True
    except Exception as e:
        st.error(f"Error loading video: {e}")

# If "Proceed" is clicked, continue with processing
if proceed:
    # Trim the video/audio if needed
    if trim_option == "Yes, trim the video to 60 seconds" and audio_duration > AUDIO_LIMIT_SECONDS:
        video = video.subclip(0, AUDIO_LIMIT_SECONDS)
        audio_duration = video.audio.duration
        st.write(f"Trimmed the video/audio to {audio_duration+1} seconds.")
    
    # Save the audio as a WAV file
    audio_path = "trimmed_audio.wav"
    with st.spinner("Extracting audio..."):
        video.audio.write_audiofile(audio_path, codec='pcm_s16le', ffmpeg_params=["-ar", "44100", "-ac", "1"])

    st.write("Extracting and transcribing audio...")

    # Google Cloud Speech-to-Text Transcription
    def transcribe_audio(audio_path):
        client = speech.SpeechClient()
        with open(audio_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,  # Set to 44100 to match the WAV file's sample rate
            language_code="en-US"
        )

        response = client.recognize(config=config, audio=audio)
        transcript = " ".join([result.alternatives[0].transcript for result in response.results])
        return transcript

    try:
        transcription = transcribe_audio(audio_path)
        st.write("Original Transcription:", transcription)
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")

    # Correct the transcription using GPT-4o from Azure OpenAI with REST API
    def correct_text_gpt4o(transcript):
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_KEY
        }
        
        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Please correct the following text: {transcript}"}
            ],
            "max_tokens": 200
        }

        response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=data)
        
        if response.status_code != 200:
            st.error(f"Error {response.status_code}: {response.text}")
            return "Error in API call"

        result = response.json()
        
        if 'choices' not in result:
            st.error(f"Unexpected response format: {result}")
            return "Error: Unexpected response format"

        corrected_text = result['choices'][0]['message']['content']

        cleaned_text = re.sub(r"^.*?corrected version of the text: ", "", corrected_text, flags=re.IGNORECASE).strip()
        cleaned_text = re.sub(r"^(Sure, here is the corrected version of the text:|Here is the corrected version:|Here is the corrected text:)\s*", "", cleaned_text, flags=re.IGNORECASE).strip()
        cleaned_text = re.sub(r"^.*?corrected text:\s*", "", cleaned_text, flags=re.IGNORECASE).strip()

        return cleaned_text

    try:
        corrected_transcription = correct_text_gpt4o(transcription)
        st.write("Corrected Transcription:", corrected_transcription)
    except Exception as e:
        st.error(f"Error correcting transcription: {e}")

    # Google Cloud Text-to-Speech (using the selected voice model)
    def text_to_speech(text):
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name=voice_name,
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE if gender == "Female" else texttospeech.SsmlVoiceGender.MALE
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            effects_profile_id=["small-bluetooth-speaker-class-device"]
        )

        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        output_audio_path = "output_audio.wav"
        with open(output_audio_path, "wb") as out:
            out.write(response.audio_content)

        return output_audio_path

    try:
        tts_audio_path = text_to_speech(corrected_transcription)
        st.write("Generated new audio with corrected transcription.")
        st.audio(tts_audio_path)
    except Exception as e:
        st.error(f"Error generating audio: {e}")

    # Replace audio in the trimmed video with the new audio
    def replace_audio_in_video(video, new_audio_path, output_path):
        audio = mp.AudioFileClip(new_audio_path)
        speed_factor = video.duration / audio.duration

        # If the audio is significantly different in length, adjust the video speed
        if abs(1 - speed_factor) > 0.05:  # If the difference is more than 5%
            video = video.speedx(speed_factor)

        final_video = video.set_audio(audio)
        
        # Write the final video with synced audio
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
        final_video.close()
        audio.close()

    output_video_path = "trimmed_video_with_ai_audio.mp4"

    # Show a spinner while processing the video
    with st.spinner("Processing the final video, please wait..."):
        try:
            replace_audio_in_video(video, tts_audio_path, output_video_path)
            st.success("Final video processing complete!")
            st.write("Final trimmed video with AI-generated voice:")
            st.video(output_video_path)
        except Exception as e:
            st.error(f"Error processing final video: {e}")
