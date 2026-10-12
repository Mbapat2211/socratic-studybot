from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import json
import asyncio

from elevenlabs import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation, ConversationInitiationData

app = FastAPI()

# ── Put your ElevenLabs API key here ──────────────────────────────────────────
ELEVENLABS_API_KEY = "sk_08efe8dccd60244bbf6c9e69ff1f9eb72d91308f7f758a1c"
# ──────────────────────────────────────────────────────────────────────────────

# v2.0 Anti-Loop Socratic System Guardrail
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
    # 1. Dynamically rebuild context from active/enabled files (sorted by ID for stable prefix caching)
    active_files = [f for f in req.workspace_files if f.enabled]
    active_files.sort(key=lambda x: x.id)

    document_text = "\n\n".join(
        [f"--- FILE: {f.filename} ---\n{f.text}" for f in active_files]
    ) if active_files else "NO STUDY MATERIALS ATTACHED."

    user_payload = f"[CONTEXT FILE ATTACHMENTS]:\n{document_text}\n\n[STUDENT QUERY]: {req.student_query}"

    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

        # 2. Create a temporary text-only agent with the Socratic guardrail baked in
        agent_response = client.conversational_ai.agents.create(
            conversation_config={
                "agent": {
                    "prompt": {
                        "prompt": SYSTEM_GUARDRAIL,
                    },
                    "first_message": None,
                    "language": "en",
                },
                "conversation": {
                    "text_only": True,
                },
            },
        )
        agent_id = agent_response.agent_id

        try:
            # 3. Run a text-only conversation session
            result_holder = {}

            def on_agent_response(text: str):
                result_holder["raw"] = result_holder.get("raw", "") + text

            config = ConversationInitiationData(
                conversation_config_override={
                    "conversation": {"text_only": True}
                }
            )

            conversation = Conversation(
                client=client,
                agent_id=agent_id,
                requires_auth=True,
                config=config,
                callback_agent_response=on_agent_response,
            )

            conversation.start_session()
            conversation.send_user_message(user_payload)

            # Wait for the agent response to arrive
            await asyncio.sleep(5)
            conversation.end_session()

            raw_text = result_holder.get("raw", "")
            if not raw_text:
                raise HTTPException(status_code=500, detail="No response received from ElevenLabs agent.")

            # 4. Strip any markdown fences and parse JSON
            clean = raw_text.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            clean = clean.strip()

            return json.loads(clean)

        finally:
            # 5. Clean up the ephemeral agent
            try:
                client.conversational_ai.agents.delete(agent_id=agent_id)
            except Exception:
                pass

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
