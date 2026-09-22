import json
import os
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def _llm():
    # Add OPENAI_API_KEY in Streamlit Secrets or your local environment.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def extract_laundry_info(text: str) -> Dict[str, Any]:
    llm = _llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You extract structured laundry information from natural language.
Return ONLY valid JSON with these keys:
load_kg (number from 0 to 10),
dirt_level (integer 0 to 100),
water_saving (integer 0 to 100),
fabric (short string).

Interpret natural language reasonably. If a value is not provided, use a sensible
default: load_kg=5, dirt_level=50, water_saving=50, fabric='Mixed'.
Do not include markdown or extra text."""
        ),
        ("human", "{user_text}")
    ])

    chain = prompt | llm
    response = chain.invoke({"user_text": text})

    content = response.content
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    data = json.loads(content)
    data["load_kg"] = max(0.0, min(10.0, float(data["load_kg"])))
    data["dirt_level"] = max(0, min(100, int(data["dirt_level"])))
    data["water_saving"] = max(0, min(100, int(data["water_saving"])))
    data["fabric"] = str(data["fabric"])
    return data


def explain_result(user_text: str, info: Dict[str, Any], result: Dict[str, Any]) -> str:
    llm = _llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are explaining a student fuzzy-logic laundry recommendation.
Use the supplied result; do not change it. Explain in 3-5 simple sentences.
Mention that the final setting comes from the fuzzy inference system."""
        ),
        (
            "human",
            """Original user description:
{user_text}

Extracted values:
{info}

Fuzzy result:
{result}"""
        )
    ])

    chain = prompt | llm
    response = chain.invoke({
        "user_text": user_text,
        "info": json.dumps(info),
        "result": json.dumps({
            "wash_intensity": result["wash_intensity"],
            "wash_time": result["wash_time"],
            "water_level": result["water_level"],
            "spin": result["spin"],
            "crisp_intensity": result["crisp_intensity"],
        }),
    })
    return response.content
