from openai import OpenAI
import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime


class DirectPromptAgent:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=openai_api_key, base_url="https://openai.vocareum.com/v1")

    def respond(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


class AugmentedPromptAgent:
    def __init__(self, openai_api_key: str, persona: str):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.client = OpenAI(api_key=openai_api_key, base_url="https://openai.vocareum.com/v1")

    def respond(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Forget any pervious conversational context. You are {self.persona}."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


class KnowledgeAugmentedPromptAgent:
    def __init__(self, openai_api_key: str, persona: str, knowledge: str):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.knowledge = knowledge
        self.client = OpenAI(api_key=openai_api_key, base_url="https://openai.vocareum.com/v1")

    def respond(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona} knowledge based assistant. Forget all previous context. Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge}.Answer the prompt based on this knowledge, not your own."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    

class EvaluationAgent:
    def __init__(self, openai_api_key: str, persona: str, evaluation_criteria: str, worker_agent, max_interactions: int):
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker = worker_agent
        self.max_interactions = max_interactions
        self.client = OpenAI(api_key=openai_api_key, base_url="https://openai.vocareum.com/v1")

    def respond(self, prompt: str) -> dict:
        iterations = 0
        current_prompt = prompt

        for _ in range(self.max_interactions):
            iterations += 1
            worker_response = self.worker.respond(current_prompt)

            evaluation = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a strict evaluator. Reply with only 'PASS' or 'FAIL'."},
                    {"role": "user", "content": f"Criteria: {self.evaluation_criteria}\nResponse: {worker_response}"}
                ]
            ).choices[0].message.content.strip()

            if evaluation == "PASS":
                break

            current_prompt = f"Improve this response to meet the criteria '{self.evaluation_criteria}': {worker_response}"

        return {"response": worker_response, "evaluation": evaluation, "iterations": iterations}
    

class RoutingAgent:
    def __init__(self, openai_api_key: str, agents: str):
        self.openai_api_key = openai_api_key
        self.agents = agents
        self.client = OpenAI(api_key=openai_api_key, base_url="https://openai.vocareum.com/v1")

    def get_embedding(self, text: str) -> list:
        return self.client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    ).data[0].embedding
    

    def respond(self, prompt: str) -> str:
        prompt_embedding = self.get_embedding(prompt)
        
        best_agent = None
        best_score = -1
        
        for agent_info in self.agents:
            agent_embedding = self.get_embedding(agent_info["description"])
            score = sum(a * b for a, b in zip(prompt_embedding, agent_embedding))
            if score > best_score:
                best_score = score
                best_agent = agent_info["func"]
        
        return best_agent(prompt)
    

class ActionPlanningAgent:
    def __init__(self, openai_api_key: str, knowledge: str):
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge
        self.client = OpenAI(api_key=openai_api_key, base_url="https://openai.vocareum.com/v1")

    def respond(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are an Action Planning Agent that extracts steps using the provided knowledge {self.knowledge}."},
                {"role": "user", "content": prompt}
            ]
        )
        lines = response.choices[0].message.content.splitlines()
        actions = [line.strip() for line in lines if line.strip()]
        return actions
    

# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })

            start = end - self.chunk_overlap
            chunk_id += 1

        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
            ],
            temperature=0
        )

        return response.choices[0].message.content