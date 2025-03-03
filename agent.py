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
            prompt = f"Generate novel hypotheses for: {research_goal.goal}. Constraints: {research_goal.constraints}. Preferences: {research_goal.preferences}."
            response = model.generate_content(prompt)
            hypotheses = []
            for i, idea in enumerate(response.candidates[0].content.parts):
                hypothesis = Hypothesis(
                    id=f"hypothesis_{i}",
                    content=idea.text,
                    novelty_score=0.8,  # Placeholder, replace with actual scoring logic
                    feasibility_score=0.7,  # Placeholder, replace with actual scoring logic
                    safety_score=0.9  # Placeholder, replace with actual scoring logic
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

async def main_workflow(research_goal: ResearchGoal):
    generation_agent = GenerationAgent()
    reflection_agent = ReflectionAgent()

    hypotheses = await generation_agent.generate_hypotheses(research_goal)
    reviewed_hypotheses = await reflection_agent.review_hypotheses(hypotheses)
    return reviewed_hypotheses

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
    st.write("Enter your research goal and constraints to generate hypotheses.")

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
            reviewed_hypotheses = asyncio.run(main_workflow(research_goal))
            if reviewed_hypotheses:
                st.write("### Hypotheses")
                display_hypotheses(reviewed_hypotheses)
            else:
                st.warning("No hypotheses generated. Please check your input and try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
