from workflow_agents.base_agent import DirectPromptAgent
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the Capital of France?"

direct_agent = DirectPromptAgent(openai_api_key=openai_api_key)
direct_agent_response = direct_agent.respond(prompt)

print(direct_agent_response)

print("General knowledge from gpt-3.5-turbo.")

