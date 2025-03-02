# AI-Agent
AI Agent assignment for iMAX intern

LOOM VIDEO LINK :https://drive.google.com/file/d/1ictQwscsrknKrQdw7-Vh3omDACWNl9JU/view?usp=sharing

AI Agent for Cold Calling in Hinglish
Project Description
This project implements an AI-powered conversational agent designed to conduct cold calls in Hinglish (a blend of Hindi and English) for three specific business scenarios:

Demo Scheduling: Schedules product demos for an ERP system.
Candidate Interviewing: Conducts initial screening interviews with job applicants.
Payment/Order Follow-up: Reminds or requests customers to release payments or place orders.
The agent leverages natural language processing (NLP) and speech technologies to understand user input, classify intents, maintain conversation context, and respond in a human-like manner. It integrates advanced models like IndicBERT for intent classification and the Groq API for response generation, ensuring efficient and context-aware interactions.


How to use this code:
1.Clone the Repository
git clone <repository-url>
cd <repository-folder>

2.Install Dependencies using the requirments.txt or install langchain langchain-groq transformers torch speechrecognition gtts python-dotenv manually
3.Install mpg123 for Audio Playback
4.Set Up Environment Variables: Create a .env file in the project root with your Groq API key
5.Activate the virtual enviornment 
6.Run the code using python ai_agent.py

## When prompted, enter a use case: demo, interview, or payment.
## Speak to interact with the agent; say "bye" or "exit" to end the call.

MODELS
1.IndicBert
2.Groq LLM
3.No custom dataset was used due to time constrains

Architecture Overview
The agent follows a modular architecture:

1.Input Layer: Captures spoken input via microphone and converts it to text.
2.Processing Layer: Analyzes input for intent and generates responses using NLP models.
3.Output Layer: Converts text responses to speech for playback.
4.State Management: Tracks conversation history to maintain context.

Key Components
Speech Input (SpeechRecognition):
Uses the speechrecognition library with Google Speech API to convert audio to text.
Configured for Hindi (language="hi-IN") to handle Hinglish input.
Intent Classification (IndicBERT):
Processes text input to classify user intent (e.g., agree_to_demo, provide_info).
Implemented with Hugging Face’s transformers pipeline.
Response Generation (Groq + LangChain):
Groq API: Generates dynamic Hinglish responses based on a prompt that includes use case, intent, history, and user input.
LangChain: Manages the conversation flow via LLMChain and maintains context with ConversationBufferMemory.
Speech Output (gTTS):
Converts text responses to speech using Google Text-to-Speech (gTTS) with a Hindi accent (lang="hi").
Playback is handled by mpg123, a lightweight audio player.
Conversation Loop:
Starts with an initial message based on the use case.
Continuously listens, processes input, and responds until the user says "bye" or "exit".
Ensures proper termination by breaking the loop on exit commands.


## We can finetune the indicbert with our own customized dataset
