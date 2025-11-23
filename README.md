# HAMBOT
Ham Radio AI Bot

**Overview**  
Welcome to HAMBOT, an AI-powered bot that blends the traditional world of amateur radio with cutting-edge artificial intelligence! At its core, HAMBOT is designed to receive, and transmit, messages using CW Morse Code. This project is the first step in merging radio communications with AI, and it's just the beginning of what's possible!

**Key Features**  
*Morse Code Reception: HAMBOT receives Morse code signals using the PZSDR board and processes them using GnuRadio on a DragonOS Linux distribution.  
*Audio Feature Extraction: Python and Librosa are used to extract features from incoming Morse code audio signals.  
*Morse Code Decoding: TensorFlow helps classify and decode the Morse code messages into readable text.  
*Natural Language Processing: Powered by Ollama’s LLM, HAMBOT can process and respond with conversational AI capabilities.  
*Future Voice Interaction: Currently, HAMBOT is being enhanced to understand and speak via voice messages, leveraging Automatic Speech Recognition (ASR) tools.  

**How It Works**  
Hardware Setup: The PZSDR board is used for RF reception, and GnuRadio is installed on the DragonOS Linux distribution to handle the communication interface.  
Signal Processing: Librosa processes the incoming audio signals from Morse code, extracting relevant features.  
Morse Code Decoding: TensorFlow is used to classify and decode the audio features into Morse code text.  
AI Brain: The decoded messages are passed to Ollama's LLM, which is the conversational AI that powers HAMBOT’s responses.  
Voice Messages (In Development): Currently working on adding speech-to-text and text-to-speech capabilities to allow voice-based communication.  

**Current Status**  
Morse Code Reception and Decoding: Fully functional and operational.  
Conversation AI (NLP): Integrated and working with Ollama's LLM for natural language processing.  
Voice Interaction: In progress. HAMBOT will soon be able to interact with users via voice messages.  
