import os
import io
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from scipy.io.wavfile import write
import random
import sys

# -------------------------- MORSE CODE DICTIONARY --------------------------
MORSE_CODE = {
    'a': '.-',    'b': '-...',  'c': '-.-.', 'd': '-..',  'e': '.',    'f': '..-.',
    'g': '--.',   'h': '....',  'i': '..',   'j': '.---', 'k': '-.-',  'l': '.-..',
    'm': '--',    'n': '-.',    'o': '---',  'p': '.--.', 'q': '--.-', 'r': '.-.',
    's': '...',   't': '-',     'u': '..-',  'v': '...-', 'w': '.--',  'x': '-..-',
    'y': '-.--',  'z': '--..',
    '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
    '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
    ' ': '       '   # 7 units space for word separation
}

# -------------------------- FEATURE EXTRACTION --------------------------
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=7, n_fft=512)
    return np.mean(mfcc, axis=1)

# -------------------------- LOAD TRAINING DATA --------------------------
def load_data_from_directory():
    features = []
    labels = []
    directory_paths = [
        'data/morse1',
        'data/morse2',
        'data/morse3',
        'data/morse4',
        'data/morse5',
        'data/morse6',
        'data/morse7',
        'data/morse8',
        'data/morse9',
        'data/morse10',
        'data/morse11',
        'data/morse12',
        'data/morse13',
        'data/morse14',
        'data/morse15',
        'data/morse16',
        'data/morse17',
        'data/morse18',
        'data/morse19',
        'data/morse20',
        'data/morse21',
        'data/morse22',
        'data/morse23',
        'data/morse24',
        'data/morse25',
        'data/morse26',
        'data/morse27',
        'data/morse28',
        'data/morse29',
        'data/morse30',
        'data/morse31',
        'data/morse32',
        'data/morse33',
        'data/morse34',
        'data/morse35',
        'data/morse36',
        'data/morse37',
        'data/morse38',
        'data/morse39',
        'data/morse40',
        'data/morse50',
        'data/morse51',
        'data/morse52',
        'data/morse53',
        'data/morse54',
        'data/morse55',
        'data/morse56',
        'data/morse57',
        'data/morse58',
        'data/morse59',
        'data/morse60',
        'data/morse61',
        'data/morse62',
        'data/morse63',
        'data/morse64',
        'data/morse65',
        'data/morse66',
        'data/morse67',
        'data/morse68',
        'data/morse69',
        'data/morse70',
        'data/morse71',
        'data/morse72',
        'data/morse73',
        'data/morse74',
        'data/morse75',
        'data/morse76',
        'data/morse77',
        'data/morse78',
        'data/morse79',
        'data/morse80',
        'data/morse81',
        'data/morse82',
        'data/morse83',
        'data/morse84',
        'data/morse85',
        'data/morse86',
        'data/morse87',
        'data/morse88',
        'data/morse89',
        'data/morse90',
        'data/morse91',
        'data/morse92',
        'data/morse93',
        'data/morse94',
        'data/morse95',
        'data/morse96',
        'data/morse97',
        'data/morse98',
        'data/morse99',
        'data/morse100',
        'data/morse101',
        'data/morse102',
        'data/morse103',
        'data/morse104',
        'data/morse105',
        'data/morse106',
        'data/morse107',
        'data/morse108',
        'data/morse109',
        'data/morse110',
        'data/morse111',
        'data/morse112',
        'data/morse113',
        'data/morse114',
        'data/morse115',
        'data/morse116',
        'data/morse117',
        'data/morse118',
        'data/morse119',
        'data/morse120',
        'data/morse121',
        'data/morse122',
        'data/morse123',
        'data/morse124',
        'data/morse125',
        'data/morse126',
        'data/morse127',
        'data/morse128',
        'data/morse129',
        'data/morse130',
        'data/morse131',
        'data/morse132',
        'data/morse133',
        'data/morse134',
        'data/morse135',
        'data/morse136',
        'data/morse137',
        'data/morse138',
        'data/morse139',
        'data/morse140',
        'data/morse150',
        'data/morse151',
        'data/morse152',
        'data/morse153',
        'data/morse154',
        'data/morse155',
        'data/morse156',
        'data/morse157',
        'data/morse158',
        'data/morse159',
        'data/morse160',
        'data/morse161',
        'data/morse162',
        'data/morse163',
        'data/morse164',
        'data/morse165',
        'data/morse166',
        'data/morse167',
        'data/morse168',
        'data/morse169',
        'data/morse170',
        'data/morse171',
        'data/morse172',
        'data/morse173',
        'data/morse174',
        'data/morse175',
        'data/morse176',
        'data/morse177',
        'data/morse178',
        'data/morse179',
        'data/morse180',
        'data/morse181',
        'data/morse182',
        'data/morse183',
        'data/morse184',
        'data/morse185',
        'data/morse186',
        'data/morse187',
        'data/morse188',
        'data/morse189',
        'data/morse190',
        'data/morse191',
        'data/morse192',
        'data/morse193',
        'data/morse194',
        'data/morse195',
        'data/morse196',
        'data/morse197',
        'data/morse198',
        'data/morse199',
        'data/morse200',
        'manual data/morse1',
        'manual data/morse2',
        'manual data/morse3',
        'manual data/morse4',
        'manual data/morse5',
        'manual data/morse6',
        'manual data/morse7',
        'manual data/morse8',
        'manual data/morse9',
        'manual data/morse10'
    ]
    for directory_path in directory_paths:
        for filename in os.listdir(directory_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(directory_path, filename)
                y, sr = librosa.load(file_path, sr=None)
                feature = extract_features(y, sr)
                features.append(feature)
                label = filename.split('.')[0]  # e.g. "a", "b", "0" etc.
                labels.append(label)

    return np.array(features), np.array(labels)

# -------------------------- TRAIN THE MODEL --------------------------
print("Loading training data...")
X, y = load_data_from_directory()
print(f"Loaded {len(X)} training samples")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=3)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=5)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)
print("Random Forest Test Accuracy:", accuracy_score(y_test, y_pred))

# -------------------------- DECODE INPUT MORSE.WAV --------------------------
input_file = sys.argv[1] if len(sys.argv) > 1 else "morse.wav"

if not os.path.exists(input_file):
    print(f"Error: File {input_file} not found")
    sys.exit(1)

audio = AudioSegment.from_wav(input_file)
audio = audio.set_channels(1)  # Make mono

# Tunable parameters
min_silence_len = 200      # ms
silence_thresh = -40       # dBFS
keep_silence = 1           # ms
min_chunk_len = 40         # ms
word_space_threshold = 280 # ms

print(f"\nProcessing {input_file} ...")

nonsilent_intervals = detect_nonsilent(
    audio,
    min_silence_len=min_silence_len,
    silence_thresh=silence_thresh,
    seek_step=20
)

used_intervals = [(start, end) for start, end in nonsilent_intervals if end - start >= min_chunk_len]

if not used_intervals:
    print("No Morse signals detected.")
    sys.exit(0)

chunks = []
for start, end in used_intervals:
    start_k = max(0, start - keep_silence)
    end_k = end + keep_silence
    chunk = audio[start_k:end_k]
    chunks.append(chunk)

gaps_ms = [used_intervals[i][0] - used_intervals[i-1][1] for i in range(1, len(used_intervals))]

print(f"Detected {len(chunks)} Morse letters/symbols")

# Predict each letter
predicted_letters = []
for chunk in chunks:
    buffer = io.BytesIO()
    chunk.export(buffer, format="wav")
    buffer.seek(0)
    
    y, sr = librosa.load(buffer, sr=None)
    feature = extract_features(y, sr).reshape(1, -1)
    pred_encoded = rf_classifier.predict(feature)
    letter = label_encoder.inverse_transform(pred_encoded)[0]
    predicted_letters.append(letter)

# Build decoded message with spaces
message_parts = []
for i in range(len(predicted_letters)):
    message_parts.append(predicted_letters[i])
    if len(gaps_ms) > i and gaps_ms[i] > word_space_threshold:
        message_parts.append(" ")

decoded_message = ''.join(message_parts).strip()
print("\nDecoded message:", decoded_message)

print("\nSilence gaps (ms):", gaps_ms)

# -------------------------- SEND TO OLLAMA VIA LANGCHAIN --------------------------
llm = ChatOllama(model="gemma3:27b", temperature=0.7)  # Adjust model as needed
# ChatPromptTemplate example for conversational AI
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a radio amateur operator that responses with short answers. your call sign is EP2ISA."),
    ("human", "{user_input}")
])

chat_chain = chat_template | llm

chat_response = chat_chain.invoke({"user_input": decoded_message})
ollama_response = chat_response.content.strip()
print("Chat Template Response:", ollama_response)

#response = llm.invoke(decoded_message)
#print("\nOllama response:", ollama_response)

# -------------------------- GENERATE MORSE AUDIO FOR OLLAMA RESPONSE --------------------------
# Configuration from generator.py
SAMPLE_RATE = 8000
DIT_LENGTH_MS = 80
FREQ = 700
AMPLITUDE = 0.9
ADD_NOISE = False
NOISE_LEVEL = 0.001

dit = int(SAMPLE_RATE * DIT_LENGTH_MS / 1000)
dah = dit * 3
element_space = dit
letter_space = dit * 3
word_space = dit * 7

t = np.linspace(0, DIT_LENGTH_MS/1000, dit, endpoint=False)
tone_dit = AMPLITUDE * np.sin(2 * np.pi * FREQ * t)
tone_dah = np.tile(tone_dit, 3)

def generate_morse_audio(code):
    audio = np.array([])
    parts = code.split(' ')
    
    for i, letter in enumerate(parts):
        if letter == '':  # word space
            audio = np.concatenate([audio, np.zeros(word_space)])
            continue
            
        for j, symbol in enumerate(letter):
            if symbol == '.':
                audio = np.concatenate([audio, tone_dit])
            elif symbol == '-':
                audio = np.concatenate([audio, tone_dah])
            if j < len(letter) - 1:
                audio = np.concatenate([audio, np.zeros(element_space)])
        
        if i < len(parts) - 1:
            audio = np.concatenate([audio, np.zeros(letter_space)])
    
    # fade_length = random.randint(0, 6)
    # if fade_length > 0:
        # fade = np.linspace(0, 1, fade_length)  # Fixed to fade in/out properly
        # audio[:fade_length] *= fade
        # audio[-fade_length:] *= fade[::-1]
    
    if ADD_NOISE:
        noise = NOISE_LEVEL * np.random.randn(len(audio))
        audio = audio + noise
    
    return np.int16(audio * 32767)

# Convert Ollama response to Morse code (lowercase, ignore non-supported chars)
ollama_response_clean = ''.join(c.lower() for c in ollama_response if c.lower() in MORSE_CODE)
morse_code_str = ' '.join(MORSE_CODE.get(c, '') for c in ollama_response_clean)

output_audio = generate_morse_audio(morse_code_str)

output_file = "output_morse.wav"
write(output_file, SAMPLE_RATE, output_audio)
print(f"\nGenerated Morse audio for Ollama response saved to: {output_file}")

# -------------------------- DECODE OUTPUT MORSE.WAV --------------------------
input_file = "output_morse.wav"

if not os.path.exists(input_file):
    print(f"Error: File {input_file} not found")
    sys.exit(1)

audio = AudioSegment.from_wav(input_file)
audio = audio.set_channels(1)  # Make mono

# Tunable parameters
min_silence_len = 200      # ms
silence_thresh = -40       # dBFS
keep_silence = 0           # ms
min_chunk_len = 40         # ms
word_space_threshold = 280 # ms

print(f"\nProcessing {input_file} ...")

nonsilent_intervals = detect_nonsilent(
    audio,
    min_silence_len=min_silence_len,
    silence_thresh=silence_thresh,
    seek_step=20
)

used_intervals = [(start, end) for start, end in nonsilent_intervals if end - start >= min_chunk_len]

if not used_intervals:
    print("No Morse signals detected.")
    sys.exit(0)

chunks = []
for start, end in used_intervals:
    start_k = max(0, start - keep_silence)
    end_k = end + keep_silence
    chunk = audio[start_k:end_k]
    chunks.append(chunk)

gaps_ms = [used_intervals[i][0] - used_intervals[i-1][1] for i in range(1, len(used_intervals))]

print(f"Detected {len(chunks)} Morse letters/symbols")

# Predict each letter
predicted_letters = []
for chunk in chunks:
    buffer = io.BytesIO()
    chunk.export(buffer, format="wav")
    buffer.seek(0)
    
    y, sr = librosa.load(buffer, sr=None)
    feature = extract_features(y, sr).reshape(1, -1)
    pred_encoded = rf_classifier.predict(feature)
    letter = label_encoder.inverse_transform(pred_encoded)[0]
    predicted_letters.append(letter)

# Build decoded message with spaces
message_parts = []
for i in range(len(predicted_letters)):
    message_parts.append(predicted_letters[i])
    if len(gaps_ms) > i and gaps_ms[i] > word_space_threshold:
        message_parts.append(" ")

decoded_message = ''.join(message_parts).strip()
print("\nDecoded message:", decoded_message)

print("\nSilence gaps (ms):", gaps_ms)

