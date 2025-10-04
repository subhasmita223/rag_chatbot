import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

PROMPT_TEMPLATE = """
You are being provided with some of the data that you might don't know about.
You are tasked to understand that data and then generate a response based on user's 
query, and provide an answer based on the data provided and provide full context from the data.
If the data provided is not enough to provide a correct and accurate answer just say
that : \"The Data Provided is Insufficient To Answer Your Query\".
Remember Don't make up an answer.
If the user is greeting you then just provide a greeting response to the user.
"""

def generate_output(User_Query, Data_Chunks):
    client = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))
    prompt = f"User Query: {User_Query}\n\nDATA:\n{Data_Chunks}\n\n{PROMPT_TEMPLATE}"
    print(prompt)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents = prompt
    )
    
    return response.text





# TESTING

if __name__ == "__main__":
    # data_chunk = """
    # The monopoly position has also led to scrutiny over convenience fees , and calls for increased competition and transparency have gained traction . 16 . Security and Fraud Prevention Given its massive user base , IRCTC is a prime target for cyber threats . The organization has implemented security protocols such as CAPTCHA , OTP verification , and encrypted payment gateways . It collaborates with cybersecurity agencies to protect user data and transaction integrity . 17 . Partnerships and Collaborations IRCTC partners with numerous private players for food delivery , tourism packages , and payment gateways 
    # Introduction to IRCTC The Indian Railway Catering and Tourism Corporation ( IRCTC ) is a government-owned subsidiary of the Indian Railways , tasked with managing online ticketing , catering , and tourism operations . It is one of the most prominent public sector enterprises in India , playing a crucial role in modernizing and digitizing the experience of railway travel for millions of Indians every day . 2 . Establishment and Purpose IRCTC was established in September 1999 under the Ministry of Railways
    # Customer Service and User Experience To improve customer experience , IRCTC has integrated features like live chat , multi-language support , email alerts , and a dedicated helpline . Despite periodic criticism over ticket failures during peak times , the overall system has become more reliable , with user interface improvements and faster processing . 15 . Challenges and Criticisms Despite its success , IRCTC faces multiple challenges including server overloads during peak hours , issues with payment gateways , and complaints over food quality
    # The aim was to professionalize and streamline non-core activities of the Indian Railways , such as catering and tourism , while also pioneering e-governance initiatives . Over the years , it has grown into a major service provider , handling ticket reservations , food services , and travel packages across the country . 3 . Online Ticketing Services One of IRCTC ’ s most well-known contributions is its online ticketing platform—irctc.co.in—which revolutionized railway bookings in India . The portal allows users to book train tickets , check PNR status , seat availability , and schedule information
    # While challenges remain , its consistent innovation and adaptability position it as a model PSU , blending public service with technological advancement and commercial success . IRCTC Customer Care Number 1800-2524-51
    # """
    import Data_Ingestion as DE
    DE.ingest_documents()
    prompt = "What is Cosine Similarity"
    data_chunk = DE.query_chromadb(prompt)
    print(data_chunk)
    print(generate_output(prompt, data_chunk))
    # print(generate_output("What is NeGD", data_chunk))
    