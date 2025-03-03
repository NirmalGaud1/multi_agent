import streamlit as st
import google.generativeai as genai
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass

API_KEY = "AIzaSyA-9-lTQTWdNM43YdOXMQwGKDy0SrMwo6c"

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
        try:
            prompt = f"Generate only the individual hypotheses for: {research_goal.goal}. Constraints: {research_goal.constraints}. Preferences: {research_goal.preferences}. Do not include any introductory or summary text. Provide the hypotheses one by one."
            response = model.generate_content(prompt)
            hypotheses = []
            if response.candidates and response.candidates[0].content.parts:
                hypothesis_texts = response.candidates[0].content.parts[0].text.split('\n\n')
                for i, text in enumerate(hypothesis_texts):
                    if text.strip():  # Only add if the text isn't just whitespace
                        hypothesis = Hypothesis(
                            id=f"hypothesis_{i}",
                            content=text.strip(),
                            novelty_score=0.8,
                            feasibility_score=0.7,
                            safety_score=0.9
                        )
                        hypotheses.append(hypothesis)
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
                prompt = f"Review this hypothesis: {hypothesis.content}. Assess novelty, feasibility, and safety."
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
        prompt = f"Compare these hypotheses: 1) {h1.content} 2) {h2.content}. Which is better?"
        response = model.generate_content(prompt)
        return h1 if response.candidates[0].content.parts[0].text == "1" else h2

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
    for hypothesis in hypotheses:
        st.write(f"**Content:** {hypothesis.content}")
        st.write("---")

def main():
    st.title("AI Co-Scientist System")
    st.write("Enter your research goal and constraints to generate and rank hypotheses.")

    # Input fields
    goal = st.text_input("Research Goal", "Explore the biological mechanisms of ALS.")

    # Dropdown for safety level
    safety_level = st.selectbox(
        "Safety Level",
        options=["low", "medium", "high"],
        index=2  # Default to "high"
    )

    # Dropdown for novelty requirement
    novelty_requirement = st.selectbox(
        "Novelty Requirement",
        options=["required", "not required"],
        index=0  # Default to "required"
    )

    # Dropdown for output format
    output_format = st.selectbox(
        "Output Format",
        options=["simple", "medium", "detailed"],
        index=2  # Default to "detailed"
    )

    # Combine constraints and preferences
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
