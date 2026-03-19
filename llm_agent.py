from __future__ import annotations

import json
from typing import Optional, Tuple

from langchain_ollama import ChatOllama

from config import DEFAULT_BASE_URL, DEFAULT_MODEL, VALID_MOVES
from prompts import build_agent_prompt
from schemas import AgentDecision, AgentName, LLMTrace, SimulationState


def make_llm(model_name: str = DEFAULT_MODEL, base_url: Optional[str] = DEFAULT_BASE_URL) -> ChatOllama:
    kwargs = {
        "model": model_name,
        "temperature": 0.3,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOllama(**kwargs)


def parse_decision(text: str) -> AgentDecision:
    try:
        payload = json.loads(text)
        move = str(payload.get("move", "WAIT")).upper()
        if move not in VALID_MOVES:
            move = "WAIT"
        return AgentDecision(
            move=move,
            message=str(payload.get("message", ""))[:300],
            justification=str(payload.get("justification", ""))[:400],
        )
    except Exception:
        return AgentDecision(
            move="WAIT",
            message="I could not produce a valid structured reply this step.",
            justification="Fallback to WAIT because the response was not valid JSON.",
        )


def query_agent(agent: AgentName, state: SimulationState, llm: ChatOllama) -> Tuple[AgentDecision, LLMTrace]:
    prompt = build_agent_prompt(agent, state)
    response = llm.invoke(prompt)
    raw = response.content if isinstance(response.content, str) else str(response.content)
    decision = parse_decision(raw)
    trace = LLMTrace(
        prompt=prompt,
        raw_output=raw,
        parsed_output={
            "move": decision.move,
            "message": decision.message,
            "justification": decision.justification,
        },
    )
    return decision, trace