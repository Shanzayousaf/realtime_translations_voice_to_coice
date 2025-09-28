import os
import time
import datetime
import base64
from fastapi import FastAPI, WebSocket

from dotenv import load_dotenv
import whisper
import torch
import collections, queue
import webrtcvad
from halo import Halo
import numpy as np
import pika
import json
from constants import ExclusionCases
from helpers import wait_until
import wave
from pydub import AudioSegment
import mlflow




load_dotenv() 
app = FastAPI()


model_size_api = os.environ["size_of_whisper_model_api"]
model_size_ws = os.environ["size_of_whisper_model_ws"]
model_download_location = os.environ["download_path_whisper"]
ampq_host = os.environ["AMQP_HOST"]
ampq_port = os.environ["AMQP_PORT"]
ampq_username = os.environ["AMQP_USERNAME"]
ampq_password = os.environ["AMQP_PASSWORD"]
queue_rabbitmq = os.environ["AMQP_QUEUE"]
max_len_queue = int(os.environ["max_length_queue"])
ari_queue = os.environ["ARI_QUEUE"]
interruption_time = int(os.environ['INTERRUPTION'])


model = whisper.load_model(model_size_api, download_root=model_download_location)


credentials = pika.PlainCredentials(ampq_username,ampq_password)
connection = pika.BlockingConnection(pika.ConnectionParameters(host=ampq_host, port=ampq_port, credentials= credentials))

channel = connection.channel()

def publish_message(queue, message):
    try:
        channel.basic_publish(exchange='',
                            routing_key=queue,
                            body=json.dumps(message))
    except Exception as e1:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=ampq_host, port=ampq_port, credentials= credentials))
        channel = connection.channel()
        channel.basic_publish(exchange='',
                            routing_key=queue,
                            body=json.dumps(message))


@app.post("/whisper/audio/")
async def whisper_audio(data: dict):
    audio = data.get('audio')
    decodedData = base64.b64decode(audio)
    audiofile = ("audio"+".webm")
    with open(audiofile, 'wb') as file:
       file.write(decodedData)
    audio = whisper.load_audio(audiofile)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to("cuda")
    _, probs = model.detect_language(mel)
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    return {"text": result.text,
            "is_cuda":torch.cuda.is_available(),
            "language": max(probs, key=probs.get)}



# Initialize VAD and Whisper models
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
# this part is for vad audio stuff

class Audio(object):
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50
    CHUNK_SIZE = 1024

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, audio_data=None):
        self.buffer_queue = queue.Queue()
        self.buffer_queue.put(audio_data)
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)



class VADAudio(Audio):
    def __init__(self, aggressiveness=3, device=None, input_rate=None, audio=None):
        super().__init__(device=device, input_rate=input_rate, audio_data=audio)
        self.vad = webrtcvad.Vad(aggressiveness)
        self.ring_buffer = collections.deque(maxlen=max_len_queue)
        self.ring_buffer_unvoiced = collections.deque(maxlen=interruption_time)
        self.triggered = False

    def frame_generator(self):
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            raise Exception("Resampling required")

    def vad_collector(self, padding_ms=20, ratio=0.23, channel_id=None, 
                      frames=None,  greeting_wait_time=None, greeting_count=None, is_out_bound=None):
        if frames is None: frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        if len(frames) < 0.01:
            return None
        frame_2 = np.frombuffer(frames, dtype=np.int16)
        is_speech = self.vad.is_speech(frame_2, self.sample_rate)
       
        if not self.triggered:
            num_samples = 512  # For 16000 Hz
            audio_chunks = [frame_2[i:i + num_samples] for i in range(0, len(frame_2), num_samples)]

            # Use Silero VAD to detect voice activity in each chunk
            with torch.no_grad():
                for chunk in audio_chunks:
                    if len(chunk) < num_samples:
                        chunk = np.pad(chunk, (0, num_samples - len(chunk)), 'constant')
                    # Use Silero VAD to detect voice activity
                    vad_output = vad_model(torch.from_numpy(chunk).unsqueeze(0), 16000)
                    # If speech is detected, append it to the speech buffer
                    if vad_output.item() > 0.25:
                        self.ring_buffer.append((frames, is_speech))
                    
                       
            num_voiced = len([f for f, speech in self.ring_buffer if speech])
          
            if num_voiced > int(1000 / self.frame_duration_ms):
                print("Something being spoken", channel_id)
                message_body={"type":"STOP_PLAYING_AUDIO",
                              "data":{
                                        "channelId":channel_id
                                    }
                            }
                #publish_message(ari_queue, message_body)

            else:
                now = datetime.datetime.now()
                if not greeting_count >= 1 and is_out_bound:
                    if greeting_wait_time < now and not now >= greeting_wait_time + datetime.timedelta(0, 0.02):
                        greeting_count +=1
                        print("Play greeting .........")
                        message_body = {"type":"WHISPER_TRANSCRIBE_TEXT", 
                                            "data":{"text":'',
                                            "channelId":channel_id}}
                        #publish_message(queue_rabbitmq, message_body)
                        yield greeting_count
                
            if num_voiced > ratio * self.ring_buffer.maxlen:
              
                self.triggered = True
                print("Something being spoken ", channel_id)
                message_body={"type":"STOP_PLAYING_AUDIO",
                              "data":{
                                        "channelId":channel_id
                                    }
                            }
                #publish_message(ari_queue, message_body)
                for f, s in self.ring_buffer:
                    yield f
                self.ring_buffer.clear()

        else:
            #yield frames
            self.ring_buffer_unvoiced.append((frames, is_speech))
            num_unvoiced = len([f for f, speech in self.ring_buffer_unvoiced if not speech])
           
            if num_unvoiced > 0.3 * self.ring_buffer_unvoiced.maxlen:
                self.triggered = False
                yield None
                self.ring_buffer_unvoiced.clear()

def Int2Float(sound):
    _sound = np.copy(sound)
    abs_max = np.abs(_sound).max()
    _sound = _sound.astype('float32')
    if abs_max > 0:
        _sound *= 1/abs_max
    audio_float32 = torch.from_numpy(_sound.squeeze())
    return audio_float32


model_ = whisper.load_model(model_size_ws, download_root=model_download_location)
DEFAULT_SAMPLE_RATE = 16000
        
mlflow.set_tracking_uri("http://127.0.0.1:8000")  # Or use a server URI
mlflow.set_experiment("Whisper-ASR-Experiment")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, channelId: str,
                              langCode:str, greetingWaitTime:int, isOutbound:bool):
    await websocket.accept()
    recording = bytearray()
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model="silero_vad")
    greeting_start_time = datetime.datetime.now()
    greeting_time = greeting_start_time + datetime.timedelta(0, greetingWaitTime)
    complete_sentence = []
    response_sent = False  # Variable to track if response is sent
    last_speech_time = datetime.datetime.now()
    (get_speech_ts, _, _, _, _) = utils
    vad_audio = VADAudio(aggressiveness=3, device=None, input_rate=DEFAULT_SAMPLE_RATE)
    print("Listening (ctrl-C to exit)...")
    spinner = Halo(spinner='line')
    is_speaking = False
    wav_data = bytearray()
    audio_data = bytearray()
    greeting_count = 0
    file_path = f"{channelId}.mp3"
    

    try:
        async for data in websocket.iter_bytes():
            audio_buffer = bytearray(data)
            audio_data.extend(data)
            if isinstance(data, bytes):
                recording.extend(data)
            frames = vad_audio.vad_collector(frames=audio_buffer,
                                             channel_id=channelId,
                                             greeting_wait_time=greeting_time,
                                             greeting_count=greeting_count,
                                             is_out_bound=isOutbound)

            count = 0
            for frame in frames:
                if isinstance(frame, int):
                    greeting_count = frame
                else:
                    if frame is not None:
                        if spinner:
                            spinner.start()
                        wav_data.extend(frame)
                        count += 1
                          # Update last speech time
                        response_sent = False 
                        is_speaking = True
                    else:
                        is_speaking = False
                        if spinner:
                            spinner.stop()
                        if len(wav_data) > 0:
                            start_time = time.time()
                            
                            # MLflow Logging
                            
                            newsound = np.frombuffer(audio_data, np.int16)
                            audio_float32 = Int2Float(newsound)
                            time_stamps = get_speech_ts(audio_float32, model)
                            if len(time_stamps) > 0:
                                message_body = {"type": "STOP_PLAYING_AUDIO",
                                                "data": {
                                                    "channelId": channelId
                                                }
                                                }
                                publish_message(ari_queue, message_body)
                                print("Silero VAD has detected a possible speech", time_stamps)
                                audio = whisper.pad_or_trim(audio_float32)

                                if model_size_ws == 'large-v3':
                                    mel = whisper.log_mel_spectrogram(audio, n_mels=128)
                                else:
                                    mel = whisper.log_mel_spectrogram(audio)

                                options = whisper.DecodingOptions(fp16=False,
                                                                  language=langCode.split('-')[0], )
                                result = whisper.decode(model_, mel, options)
                                
                                print("TEXT: ", " ".join(complete_sentence) + " " + result.text)
                                latency = time.time() - start_time
                                mlflow.log_param("model_size", model_size_api)
                                mlflow.log_param("language_detected", langCode)
                                mlflow.log_metric("latency_seconds", latency)
                                mlflow.log_param("cuda_available", torch.cuda.is_available())
                                mlflow.log_text(result.text, "transcription.txt")

                                is_complete = False
                                last_speech_time = datetime.datetime.now()
                                if not is_complete:
                                    complete_sentence.append(result.text)
                                else:
                                    final_text = " ".join(complete_sentence) + " " + result.text
                                    complete_sentence = []
                                    response_sent = True  # Mark response as sent

                                    now = datetime.datetime.now()
                                    if not now >= greeting_start_time + datetime.timedelta(0, 60) and greeting_count < 1 and isOutbound:
                                        greeting_count += 1
                                        print("Play greeting when someone is saying .........")
                                        message_body = {"type": "WHISPER_TRANSCRIBE_TEXT",
                                                        "data": {"text": '',
                                                                 "channelId": channelId}}
                                        #publish_message(queue_rabbitmq, message_body)
                                    
                                        
                            else:
                                print("Silero VAD has not detected... still listening....")

                            wav_data = bytearray()
                            audio_data = bytearray()

            # Fallback mechanism: if no speech is detected and response is not sent
            now = datetime.datetime.now()
            
            if (now - last_speech_time).total_seconds() > 0 and not response_sent and complete_sentence and not is_speaking:
                final_text = " ".join(complete_sentence)
                print(f"Fallback triggered: sending accumulated text - {final_text}")
                hallucination = ExclusionCases(final_text, langCode.split('-')[0]).check_hallucination()
                if not hallucination:
                    message_body = {"type": "WHISPER_TRANSCRIBE_TEXT",
                                    "data": {"text": final_text.upper(),
                                                "channelId": channelId}}
                    publish_message(queue_rabbitmq, message_body)
                else:
                    print("Hallucination detected")
                response_sent = True  # Mark response as sent
                complete_sentence = []  # Clear buffer
                
    except Exception as e:
        print(f"Error: {e}")

    finally:
        if recording:
        # Convert the raw audio data to MP3
            audio_segment = AudioSegment(
                data=bytes(recording),
                sample_width=2,  # 16-bit audio
                frame_rate=16000,  # Set based on client input
                channels=1
            )
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            
            print(f"Audio saved as {file_path} (MP3 format)")




            