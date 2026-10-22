import json
import os
import re
from typing import List

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# ── NVIDIA NIM (free tier) ─────────────────────────────────────────────────────
# Get a free key at https://build.nvidia.com (Settings → API Keys), then set it as
# NVIDIA_API_KEY in your Vercel project's Environment Variables (and a local .env).
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b")
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_GUARDRAIL = """
You are a universal cognitive scaffolding engine. Your sole objective is to train students in domain-specific problem-solving strategies, encouraging self-directed thought and minimizing AI reliance.
Never solve the problem, complete calculations, write code/essays, or provide follow-up prompts for the user to type back into an LLM.

Respond ONLY with a valid JSON object matching this structure:
{
  "textual_focus": "Cite 1-2 exact line numbers, quotes, or sections from the provided text for the student to look up in their copy.",
  "thinking_framework": "Explain ONE mental technique or analytical strategy a domain scholar uses to approach this issue.",
  "offline_exercise": "Provide ONE specific, concrete task for the student to perform ON PAPER or in their notebook before forming their argument."
}
"""

REQUIRED_KEYS = ("textual_focus", "thinking_framework", "offline_exercise")


class WorkspaceFile(BaseModel):
    id: str
    filename: str
    text: str
    enabled: bool


class SocraticRequest(BaseModel):
    student_query: str
    workspace_files: List[WorkspaceFile]


def extract_json(raw: str) -> dict:
    """Pull the JSON object out of a model reply (handles <think> blocks and code fences)."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("Model did not return a JSON object.")
    data = json.loads(cleaned[start : end + 1])
    missing = [k for k in REQUIRED_KEYS if not isinstance(data.get(k), str)]
    if missing:
        raise ValueError(f"Model reply is missing fields: {', '.join(missing)}")
    return {k: data[k] for k in REQUIRED_KEYS}


@app.post("/api/scaffold")
async def generate_scaffold(req: SocraticRequest):
    if not NVIDIA_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="NVIDIA_API_KEY is not set. Add it in your Vercel project's Environment Variables.",
        )

    active_files = [f for f in req.workspace_files if f.enabled]
    active_files.sort(key=lambda x: x.id)

    document_text = "\n\n".join(
        [f"--- FILE: {f.filename} ---\n{f.text}" for f in active_files]
    ) if active_files else "NO STUDY MATERIALS ATTACHED."

    user_payload = f"[CONTEXT FILE ATTACHMENTS]:\n{document_text}\n\n[STUDENT QUERY]: {req.student_query}"

    body = {
        "model": NVIDIA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_GUARDRAIL},
            {"role": "user", "content": user_payload},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=55) as client:
            resp = await client.post(NVIDIA_URL, json=body, headers=headers)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="The model took too long to respond. Try again.")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Could not reach NVIDIA API: {e}")

    if resp.status_code == 429:
        raise HTTPException(status_code=429, detail="NVIDIA free-tier rate limit hit. Wait a moment and retry.")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"NVIDIA API error {resp.status_code}: {resp.text[:300]}")

    try:
        content = resp.json()["choices"][0]["message"]["content"] or ""
        return extract_json(content)
    except (KeyError, IndexError, ValueError) as e:
        raise HTTPException(status_code=502, detail=f"Unexpected model response: {e}")
