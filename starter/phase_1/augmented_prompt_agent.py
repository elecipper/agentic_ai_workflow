from workflow_agents.base_agent import AugmentedPromptAgent
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

augmented_agent = AugmentedPromptAgent(openai_api_key=openai_api_key, persona=persona)

augmented_agent_response = augmented_agent.respond(prompt)

print(augmented_agent_response)

print("General knowdledge used from GPT 3.5 Turbo")
print("The persona influenced the formatting, tone, etc of the response")