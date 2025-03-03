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
                f"Generate 3 concise research hypotheses for: {research_goal.goal}. "
                f"Constraints: {research_goal.constraints}. "
                f"Preferences: {research_goal.preferences}. "
                "For each hypothesis, provide:\n"
                "Hypothesis Statement: [Your Hypothesis Statement]\n"
                "Description: [A brief description of the research]"
            )
            response = model.generate_content(prompt)
            hypotheses = []
            parts = re.split(r"Hypothesis \d+:\n\*\*", response.text)
            parts = parts[1:]

            for i, part in enumerate(parts):
                try:
                    aim_match = re.search(r"\*\*(.+?)\nDescription:", part, re.DOTALL)
                    description_match = re.search(r"Description:\s*(.+)", part, re.DOTALL)

                    if aim_match and description_match:
                        aim = aim_match.group(1).strip()
                        description = description_match.group(1).strip()

                        hypothesis = Hypothesis(
                            id=f"hypothesis_{i}",
                            aim=aim,
                            objectives=[description],
                            algorithm="N/A",
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
            f"1) Hypothesis Statement: {h1.aim}, Description: {h1.objectives[0]}. "
            f"2) Hypothesis Statement: {h2.aim}, Description: {h2.objectives[0]}. "
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
        st.write(f"**Hypothesis Statement:** {hypothesis.aim}")
        st.write(f"**Description:**")
        st.write(f"{hypothesis.objectives[0]}")
        st.write(f"**Algorithm:** {hypothesis.algorithm}")
        st.write(f"**Novelty Score:** {hypothesis.novelty_score}")
        st.write(f"**Feasibility Score:** {hypothesis.feasibility_score}")
        st.write(f"**Safety Score:** {hypothesis.safety_score}")
        st.write("---")

def main():
    st.title("AI Co-Scientist System")
    st.write("Enter your research goal and constraints to generate and rank hypotheses.")

    goal = st.text_input("Research Goal", "Explore the ethical implications of AI in autonomous vehicles.")

    safety_level = st.selectbox(
        "Safety Level",
        options=["low", "medium", "high"],
        index=2
    )

    novelty_requirement = st.selectbox(
        "Novelty Requirement",
        options=["required", "not required"],
        index=0
    )

    output_format = st.selectbox(
        "Output Format",
        options=["simple", "medium", "detailed"],
        index=2
    )

    constraints = f"safety: {safety_level}, novelty: {novelty_requirement}"
    preferences = f"format: {output_format}"

    if st.button("Generate Hypotheses"):
        research_goal = ResearchGoal(
            goal=goal,
            constraints=constraints,
            preferences=preferences
        )
        try:
            ranked_hypotheses = asyncio.run(main_workflow(research_goal))
            if ranked_hypotheses:
                st.write("### Ranked Hypotheses")
                display_hypotheses(ranked_hypotheses)
            else:
                st.warning("No hypotheses generated. Please check your input and try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
