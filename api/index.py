from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import json
import os

import google.generativeai as genai

app = FastAPI()

# ── API Keys ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")   # Get free key at aistudio.google.com
ELEVENLABS_API_KEY = "sk_08efe8dccd60244bbf6c9e69ff1f9eb72d91308f7f758a1c"   # Reserved for TTS features later
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

class WorkspaceFile(BaseModel):
    id: str
    filename: str
    text: str
    enabled: bool

class SocraticRequest(BaseModel):
    student_query: str
    workspace_files: List[WorkspaceFile]


@app.post("/api/scaffold")
async def generate_scaffold(req: SocraticRequest):
    active_files = [f for f in req.workspace_files if f.enabled]
    active_files.sort(key=lambda x: x.id)

    document_text = "\n\n".join(
        [f"--- FILE: {f.filename} ---\n{f.text}" for f in active_files]
    ) if active_files else "NO STUDY MATERIALS ATTACHED."

    user_payload = f"[CONTEXT FILE ATTACHMENTS]:\n{document_text}\n\n[STUDENT QUERY]: {req.student_query}"

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-3.8-flash",
            system_instruction=SYSTEM_GUARDRAIL,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        response = model.generate_content(user_payload)
        return json.loads(response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
