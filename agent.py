import streamlit as st
import google.generativeai as genai
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
import re

API_KEY = "YOUR_API_KEY"  # Replace with your actual API key

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
    title: str
    description: str
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
                f"Generate 3 novel research hypotheses for: {research_goal.goal}. "
                f"Constraints: {research_goal.constraints}. "
                "For each hypothesis, provide:\n"
                "Title: [Title of the hypothesis]\n"
                "Description: [A brief description of the hypothesis]\n"
                "Aim: [A clear aim of the hypothesis.]\n"
                "Objectives: [List 2-3 specific objectives.]\n"
                "Algorithm: [A detailed algorithm in numbered points.]\n"
            )
            response = model.generate_content(prompt)
            hypotheses = []
            parts = re.split(r"Hypothesis \d+:\n", response.text)
            parts = parts[1:]

            for i, part in enumerate(parts):
                try:
                    title_match = re.search(r"Title:\s*(.+?)\nDescription:", part, re.DOTALL)
                    description_match = re.search(r"Description:\s*(.+?)\nAim:", part, re.DOTALL)
                    aim_match = re.search(r"Aim:\s*(.+?)\nObjectives:", part, re.DOTALL)
                    objectives_match = re.search(r"Objectives:\s*(.+?)\nAlgorithm:", part, re.DOTALL)
                    algorithm_match = re.search(r"Algorithm:\s*(.+)", part, re.DOTALL)

                    if title_match and description_match and aim_match and objectives_match and algorithm_match:
                        title = title_match.group(1).strip()
                        description = description_match.group(1).strip()
                        aim = aim_match.group(1).strip()
                        objectives = [obj.strip() for obj in objectives_match.group(1).split("\n")]
                        algorithm = algorithm_match.group(1).strip()

                        hypothesis = Hypothesis(
                            id=f"hypothesis_{i}",
                            title=title,
                            description=description,
                            aim=aim,
                            objectives=objectives,
                            algorithm=algorithm,
                            novelty_score=0.8,
                            feasibility_score=0.7,
                            safety_score=0.9,
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

class ReflectionAgent:
    async def review_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        reviewed_hypotheses = []
        for hypothesis in hypotheses:
            try:
                prompt = (
                    f"Review this hypothesis: Title: {hypothesis.title}, Description: {hypothesis.description}, "
                    f"Aim: {hypothesis.aim}, Objectives: {hypothesis.objectives}, Algorithm: {hypothesis.algorithm}. "
                    "Assess novelty, feasibility, and safety."
                )
                response = model.generate_content(prompt)
                review = response.candidates[0].content.parts[0].text
                hypothesis.novelty_score = 0.8  # Placeholder, replace with actual scoring logic
                hypothesis.feasibility_score = 0.7  # Placeholder, replace with actual scoring logic
                hypothesis.safety_score = 0.9  # Placeholder, replace with actual scoring logic
                reviewed_hypotheses.append(hypothesis)
            except Exception as e:
                st.error(f"Error reviewing hypothesis {hypothesis.id}: {e}")
        context_memory["reviewed_hypotheses"] = reviewed_hypotheses
        return reviewed_hypotheses

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
            f"1) Title: {h1.title}, Description: {h1.description}, Aim: {h1.aim}, Objectives: {h1.objectives}, Algorithm: {h1.algorithm}. "
            f"2) Title: {h2.title}, Description: {h2.description}, Aim: {h2.aim}, Objectives: {h2.objectives}, Algorithm: {h2.algorithm}. "
            "Which is better, 1 or 2?"
        )
        response = model.generate_content(prompt)
        try:
            if "1" in response.candidates[0].content.parts[0].text:
                return h1
            elif "2" in response.candidates[0].content.parts[0].text:
                return h2
            else:
                return h1
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
    reflection_agent = ReflectionAgent()
    ranking_agent = RankingAgent()

    hypotheses = await generation_agent.generate_hypotheses(research_goal)
    reviewed_hypotheses = await reflection_agent.review_hypotheses(hypotheses)
    ranked_hypotheses = await ranking_agent.rank_hypotheses(reviewed_hypotheses)
    return ranked_hypotheses

def display_hypotheses(hypotheses: List[Hypothesis]):
    for i, hypothesis in enumerate(hypotheses):
        st.write(f"### Hypothesis {i + 1}")
        st.st.write(f"**Title:** {hypothesis.title}")
        st.write(f"**Description:** {hypothesis.description}")
        st.write(f"**Aim:** {hypothesis.aim}")
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
    preferences = st.text_area("Preferences", "format: detailed")

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
