import streamlit as st
import google.generativeai as genai
import os
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass

API_KEY = "AIzaSyA-9-lTQTWdNM43YdOXMQwGKDy0SrMwo6c"
genai.configure(api_key=API_KEY)

context_memory = {}

@dataclass
class ResearchGoal:
    goal: str
    constraints: Dict[str, Any]
    preferences: Dict[str, Any]

@dataclass
class Hypothesis:
    id: str
    content: str
    novelty_score: float
    feasibility_score: float
    safety_score: float

@dataclass
class ResearchOverview:
    hypotheses: List[Hypothesis]
    summary: str

class GenerationAgent:
    async def generate_hypotheses(self, research_goal: ResearchGoal) -> List[Hypothesis]:
        prompt = f"Generate novel hypotheses for: {research_goal.goal}. Constraints: {research_goal.constraints}."
        response = genai.generate_text(prompt=prompt)
        hypotheses = []
        for i, idea in enumerate(response["ideas"]):
            hypothesis = Hypothesis(
                id=f"hypothesis_{i}",
                content=idea["content"],
                novelty_score=idea["novelty_score"],
                feasibility_score=idea["feasibility_score"],
                safety_score=idea["safety_score"]
            )
            hypotheses.append(hypothesis)
        context_memory["hypotheses"] = hypotheses
        return hypotheses

class ReflectionAgent:
    async def review_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        reviewed_hypotheses = []
        for hypothesis in hypotheses:
            prompt = f"Review this hypothesis: {hypothesis.content}. Assess novelty, feasibility, and safety."
            review = genai.generate_text(prompt=prompt)
            hypothesis.novelty_score = review["novelty_score"]
            hypothesis.feasibility_score = review["feasibility_score"]
            hypothesis.safety_score = review["safety_score"]
            reviewed_hypotheses.append(hypothesis)
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
        prompt = f"Compare these hypotheses: 1) {h1.content} 2) {h2.content}. Which is better?"
        response = genai.generate_text(prompt=prompt)
        return h1 if response["winner"] == "1" else h2

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
        st.write(f"**Content:** {hypothesis.content}")
        st.write(f"**Novelty Score:** {hypothesis.novelty_score}")
        st.write(f"**Feasibility Score:** {hypothesis.feasibility_score}")
        st.write(f"**Safety Score:** {hypothesis.safety_score}")
        st.write("---")

def main():
    st.title("AI Co-Scientist System")
    st.write("Enter your research goal and constraints to generate and rank hypotheses.")

    goal = st.text_input("Research Goal", "Explore the biological mechanisms of ALS.")
    constraints = st.text_input("Constraints", "safety: high, novelty: required")
    preferences = st.text_input("Preferences", "format: detailed")

    if st.button("Generate Hypotheses"):
        research_goal = ResearchGoal(
            goal=goal,
            constraints=constraints,
            preferences=preferences
        )
        ranked_hypotheses = asyncio.run(main_workflow(research_goal))
        st.write("### Ranked Hypotheses")
        display_hypotheses(ranked_hypotheses)

if __name__ == "__main__":
    main()
