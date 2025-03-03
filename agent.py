import streamlit as st
import google.generativeai as genai
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
import re

API_KEY = "AIzaSyBsq5Kd5nJgx2fejR77NT8v5Lk3PK4gbH8"  # Replace with your actual API key

def configure_generative_model(api_key):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"Error configuring the generative model: {e}")
        return None

model = configure_generative_model(API_KEY)
if not model:
    st.stop()

context_memory = {}

@dataclass
class ResearchGoal:
    goal: str
    constraints: Dict[str, Any]
    preferences: Dict[str, Any]

@dataclass
class Hypothesis:
    id: str
    aim: str
    objectives: List[str]
    algorithm: str
    novelty_score: float
    feasibility_score: float
    safety_score: float

@dataclass
class ResearchOverview:
    hypotheses: List[Hypothesis]
    summary: str

class GenerationAgent:
    async def generate_hypotheses(self, research_goal: ResearchGoal) -> List[Hypothesis]:
        try:
            prompt = (
                f"Generate 3 detailed research hypotheses for: {research_goal.goal}. "
                f"Constraints: {research_goal.constraints}. "
                f"Preferences: {research_goal.preferences}. "
                "For each hypothesis, provide the following format:\n"
                "Hypothesis Statement: [Your Hypothesis Statement]\n"
                "Aim: [A clear aim of the hypothesis.]\n"
                "Objectives: [List 2-3 specific objectives, each on a new line.]\n"
                "Algorithm: [A detailed algorithm in numbered points.]\n"
                "Example:\n"
                "Hypothesis Statement: A new drug will reduce ALS symptoms.\n"
                "Aim: To evaluate the efficacy of the new drug.\n"
                "Objectives:\n"
                "- Measure muscle strength.\n"
                "- Assess patient quality of life.\n"
                "Algorithm:\n"
                "1. Administer the drug.\n"
                "2. Conduct regular assessments.\n"
                "3. Analyze the results."
            )
            response = model.generate_content(prompt)
            if not response.text:
              st.warning("The AI returned an empty response.")
              return []
            hypotheses = []
            parts = re.split(r"Hypothesis \d+:\n\*\*", response.text)
            parts = parts[1:]

            for i, part in enumerate(parts):
                try:
                    aim_match = re.search(r"\*\*(.+?)\nAim:", part, re.DOTALL)
                    aim_value_match = re.search(r"Aim:\s*(.+?)\nObjectives:", part, re.DOTALL)
                    objectives_match = re.search(r"Objectives:\s*(.+?)\nAlgorithm:", part, re.DOTALL)
                    algorithm_match = re.search(r"Algorithm:\s*(.+)", part, re.DOTALL)

                    if aim_match and aim_value_match and objectives_match and algorithm_match:
                        aim = aim_match.group(1).strip()
                        aim_value = aim_value_match.group(1).strip()
                        objectives = [obj.strip() for obj in objectives_match.group(1).split("\n")]
                        algorithm = algorithm_match.group(1).strip()

                        hypothesis = Hypothesis(
                            id=f"hypothesis_{i}",
                            aim=aim + ". Aim: " + aim_value,
                            objectives=objectives,
                            algorithm=algorithm,
                            novelty_score=0.8,
                            feasibility_score=0.7,
                            safety_score=0.9
                        )
                        hypotheses.append(hypothesis)
                    else:
                        st.error(f"Error parsing hypothesis {i}: Incomplete data. Raw response:\n{part}")
                except Exception as e:
                    st.error(f"Error parsing hypothesis {i}: {e}. Raw response:\n{part}")
            context_memory["hypotheses"] = hypotheses
            return hypotheses
        except Exception as e:
            st.error(f"Error generating hypotheses: {e}")
            return []
            
class RankingAgent:
    def __init__(self):
        self.elo_ratings = {}

    async def rank_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        for hypothesis in hypotheses:
            self.elo_ratings[hypothesis.id] = 1200
        for i in range(len(hypotheses)):
            for j in range(i + 1, len(hypotheses)):
                winner = self._simulate_match(hypotheses[i], hypotheses[j])
                self._update_elo_ratings(hypotheses[i], hypotheses[j], winner)
        ranked_hypotheses = sorted(hypotheses, key=lambda h: self.elo_ratings[h.id], reverse=True)
        return ranked_hypotheses

    def _simulate_match(self, h1: Hypothesis, h2: Hypothesis) -> Hypothesis:
        prompt = (
            f"Compare these hypotheses: "
            f"1) Hypothesis Statement: {h1.aim}, Objectives: {h1.objectives}, Algorithm: {h1.algorithm}. "
            f"2) Hypothesis Statement: {h2.aim}, Objectives: {h2.objectives}, Algorithm: {h2.algorithm}. "
            "Which is better, 1 or 2?"
        )
        response = model.generate_content(prompt)
        try:
          result = response.text
          if "1" in result:
              return h1
          elif "2" in result:
              return h2
          else:
              return h1 #default if the model fails to answer 1 or 2.
        except:
          return h1

    def _update_elo_ratings(self, h1: Hypothesis, h2: Hypothesis, winner: Hypothesis):
        k = 32
        r1, r2 = self.elo_ratings[h1.id], self.elo_ratings[h2.id]
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))
        if winner == h1:
            self.elo_ratings[h1.id] += k * (1 - e1)
            self.elo_ratings[h2.id] += k * (0 - e2)
        else:
            self.elo_ratings[h1.id] += k * (0 - e1)
            self.elo_ratings[h2.id] += k * (1 - e2)

async def main_workflow(research_goal: ResearchGoal):
    generation_agent = GenerationAgent()
    ranking_agent = RankingAgent()

    hypotheses = await generation_agent.generate_hypotheses(research_goal)
    ranked_hypotheses = await ranking_agent.rank_hypotheses(hypotheses)
    return ranked_hypotheses

def display_hypotheses(hypotheses: List[Hypothesis]):
    for i, hypothesis in enumerate(hypotheses):
        st.write(f"### Hypothesis {i + 1}")
        st.write(f"**Hypothesis Statement:** {hypothesis.aim.split('Aim:')[0]}")
        st.write(f"**Aim:** {hypothesis.aim.split('Aim:')[1]}")
        st.write(f"**Objectives:**")
        for obj in hypothesis.objectives:
            st.write(f"- {obj}")
        st.write(f"**Algorithm:** {hypothesis.algorithm}")
        st.write(f"**Novelty Score:** {hypothesis.novelty_score}")
        st.write(f"**Feasibility Score:** {hypothesis.feasibility_score}")
        st.write(f"**Safety Score:** {hypothesis.safety_score}")
        st.write("---")

def main():
    st.title("AI Co-Scientist System")
    st.write("Enter your research goal and constraints to generate and rank hypotheses.")

    goal = st.text_input("Research Goal", "Explore the biological mechanisms of ALS.")
    constraints = st.text_input("Constraints", "safety: high, novelty: required")
    preferences = st.text_input("Preferences", "format: detailed")  # Corrected line

    if st.button("Generate Hypotheses"):
        research_goal = ResearchGoal(
            goal=goal,
            constraints=constraints,
            preferences=preferences
        )
        try:
            ranked_hypotheses = asyncio.run(main_workflow(research_goal))
            if ranked_hypotheses:
                st.write("Ranked Hypotheses")
                display_hypotheses(ranked_hypotheses)
            else:
                st.warning("No hypotheses generated. Please check your input and try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
