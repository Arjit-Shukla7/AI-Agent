# Importing necessary libraries
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from gtts import gTTS
import speech_recognition as sr
import sys

# Setuping up environment variables from .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
# Initialize Groq LLM using the api
llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile")

# Test the initialization cause a lot of time it wasn't initializnd
print("Groq LLM initialized successfully")

# Initializing Hugging Face IndicBERT for intent classification
from transformers import AlbertTokenizer

model_name = "ai4bharat/indic-bert"
tokenizer = AlbertTokenizer.from_pretrained(model_name) #Used Autotokeniser before but it was causing a lot of errors due to clashing with AlbertTokenizer Tried forc running it also but error perssisted 

# Verifying the tokenizer
print(type(tokenizer))
print("Tokenizer loaded successfully!")


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)  # Adjust num_labels based on intents
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)  # -1 for CPU

# Defining intents just like in cold calls
INTENTS = {
    "agree_to_demo": 0,
    "decline": 1,
    "provide_info": 2,
    "confirm": 3,
    "request_change": 4
}
INTENT_LABELS = list(INTENTS.keys())

# Speech recognition and TTS setup
recognizer = sr.Recognizer()
microphone = sr.Microphone()

def recognize_speech():
    """
    Convert spoken input to text using Google's Speech Recognition API.
    Returns:
        str: Recognized text or empty string if recognition fails.
    """
    with microphone as source:
        print("Listening...")
        audio = recognizer.listen(source, timeout=5)
    try:
        text = recognizer.recognize_google(audio, language="hi-IN")
        print(f"User said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I didn't understand that.")
        return ""
    except sr.RequestError:
        print("Speech recognition service unavailable.")
        return ""

def speak(text):
    """
    Convert text to speech in Hindi accent using gTTS.
    Args:
        text (str): Text to be spoken.
    """
    tts = gTTS(text, lang='hi')
    tts.save("response.mp3")
    os.system("mpg123 response.mp3")  

# Response templates for initial messages
RESPONSE_TEMPLATES = {
    "demo": {
        "initial": "Hello, main iq Mettalika company se bol raha hoon. Kya aap hamare ERP system ka demo dekhna chahte hain?"
    },
    "interview": {
        "initial": "Hello, main iq Mettalika company se bol raha hoon. Interview start karein?"
    },
    "payment": {
        "initial": "Hello, main iq Mettalika company se bol raha hoon. Aapka payment pending hai, kab clear karenge?"
    }
}

def classify_intent(user_input):
    """Classify user intent using IndicBERT."""
    result = classifier(user_input)
    predicted_label = result[0]["label"]
    # Assuming labels are in format "LABEL_X" where X is the index
    return INTENT_LABELS[int(predicted_label.split("_")[-1])]

def run_conversation(use_case):
    """Run the conversation with the specified use case."""
    memory = ConversationBufferMemory()
    prompt = PromptTemplate(
        input_variables=["use_case", "intent", "history", "input"],
        template="""
        You are an AI agent conducting a conversation in Hinglish for {use_case}.
        The user's current intent is: {intent}
        The conversation history is:
        {history}
        User: {input}
        AI:
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)   ## Using langrun chains for interaction with LLMS

    # Start with initial message
    initial_response = RESPONSE_TEMPLATES[use_case]["initial"]
    speak(initial_response)
    memory.chat_memory.add_ai_message(initial_response)

    while True:
        user_input = recognize_speech()
        if user_input.lower() in ["bye", "goodbye", "exit"]:
            speak("Goodbye!")
            break
        if user_input:
            intent = classify_intent(user_input)
            history = memory.buffer
            response = chain.run(
                use_case=use_case,
                intent=intent,
                history=history,
                input=user_input
            )
            speak(response)
            memory.chat_memory.add_user_message(user_input)
            memory.chat_memory.add_ai_message(response)

if __name__ == "__main__":
    use_case = input("Select use case (demo/interview/payment): ").strip().lower()
    if use_case in ["demo", "interview", "payment"]:
        run_conversation(use_case)
    else:
        print("Invalid use case selected.")
        sys.exit(1)