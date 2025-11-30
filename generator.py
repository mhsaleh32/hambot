# morse_generator.py
# Run this once to generate your entire training dataset

import os
import numpy as np
#import sounddevice as sd
from scipy.io.wavfile import write
import random

# ========================= CONFIGURATION =========================
OUTPUT_ROOT = "data"          # Your main folder
SAMPLE_RATE = 8000                          # 8kHz matches your pipeline
DIT_LENGTH_MS = 80                          # Base unit in milliseconds (80ms = 15 WPM, 50ms = 24 WPM, 60ms = 20 WPM)
FREQ = 700                                  # Tone frequency in Hz (classic 600-800 Hz)
AMPLITUDE = 0.8                             # Volume (0.0 - 1.0)
ADD_NOISE = False                           # Add slight background noise (makes model more robust)
NOISE_LEVEL = 0.001

# Standard International Morse Code
MORSE_CODE = {
    'a': '.-',    'b': '-...',  'c': '-.-.', 'd': '-..',  'e': '.',    'f': '..-.',
    'g': '--.',   'h': '....',  'i': '..',   'j': '.---', 'k': '-.-',  'l': '.-..',
    'm': '--',    'n': '-.',    'o': '---',  'p': '.--.', 'q': '--.-', 'r': '.-.',
    's': '...',   't': '-',     'u': '..-',  'v': '...-', 'w': '.--',  'x': '-..-',
    'y': '-.--',  'z': '--..',
    '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
    '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
    ' ': '       '   # 7 units space for word separation (when generating words)
}

# Derived timing (standard ratios)
dit = int(SAMPLE_RATE * DIT_LENGTH_MS / 1000)
dah = dit * 3
element_space = dit                    # space between dit/dah in same letter
letter_space = dit * 3                 # space between letters
word_space = dit * 7                   # space between words

# Pre-generate sine wave for one dit (for speed)
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
            # Element space (except after last symbol of letter)
            if j < len(letter) - 1:
                audio = np.concatenate([audio, np.zeros(element_space)])
        
        # Letter space (except after last letter)
        if i < len(parts) - 1:
            audio = np.concatenate([audio, np.zeros(letter_space)])
    
    # Add tiny fade in/out to avoid clicks
    fade_length = np.random.randint(0, 6)
    if fade_length > 0:
        fade = np.linspace(0, 0, fade_length)
        audio[:fade_length] *= fade
        audio[-fade_length:] *= np.flip(fade)
    
    # Add background noise for robustness
    if ADD_NOISE:
        noise = NOISE_LEVEL * np.random.randn(len(audio))
        audio = audio + noise
    
    return np.int16(audio * 32767)  # 16-bit PCM

# ========================= GENERATE DATASET =========================

def create_dataset():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # 1. Generate individual letters (A-Z, 0-9) ? for character recognition
    letters_dir = os.path.join(OUTPUT_ROOT, "letters")
    os.makedirs(letters_dir, exist_ok=True)
    
    print("Generating individual letters (A-Z, 0-9)...")
    for char in "abcdefghijklmnopqrstuvwxyz0123456789":
        code = MORSE_CODE[char]
        audio = generate_morse_audio(code)
        filename = os.path.join(letters_dir, f"{char}.wav")
        write(filename, SAMPLE_RATE, audio)
        print(f"  Saved {filename} ({code})")

    # 2. Generate multiple variations per letter in morse1 to morse9 folders
    print("\nGenerating training variations (morse1 to morse200)...")
    for i in range(1, 201):
        folder = os.path.join(OUTPUT_ROOT, f"morse{i}")
        os.makedirs(folder, exist_ok=True)
        
        for char in "abcdefghijklmnopqrstuvwxyz0123456789":
            # Add small random timing variation (10%) for robustness
            global dit, dah, element_space, letter_space
            variation = 0.85 + random.random() * 0.30  # 85% to 115% speed
            old_dit = dit
            #dit = int(old_dit * variation)
            dah = dit * 3
            element_space = dit
            letter_space = dit * 3
            
            code = MORSE_CODE[char]
            audio = generate_morse_audio(code)
            
            # Restore original timing
            dit, dah, element_space, letter_space = old_dit, old_dit*3, old_dit, old_dit*3
            
            filename = os.path.join(folder, f"{char}.wav")
            write(filename, SAMPLE_RATE, audio)
    
    # 3. Bonus: Generate common words for testing (optional)
    words_dir = os.path.join(OUTPUT_ROOT, "words_example")
    os.makedirs(words_dir, exist_ok=True)
    
    common_words = ["hello world", "ep2isa", "cq", "iran", "test", "sos", "qrz qrz", "name", "qth"]
    print(f"\nGenerating example words in {words_dir}...")
    for word in common_words:
        code = " ".join(MORSE_CODE[c] for c in word)
        audio = generate_morse_audio(" ".join(MORSE_CODE[c] for c in word))
        filename = os.path.join(words_dir, f"{word}.wav")
        write(filename, SAMPLE_RATE, audio)
        print(f"  Saved {word}.wav")

    print("\nAll done! Your perfect training dataset is ready.")
    print(f"   ? Individual letters: {letters_dir}")
    print(f"   ? Training folders : {OUTPUT_ROOT}/morse1 to morse100")
    print(f"   ? Test words       : {words_dir}")

if __name__ == "__main__":
    create_dataset()
