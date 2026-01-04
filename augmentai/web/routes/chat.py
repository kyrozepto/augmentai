"""Chat API with WebSocket for interactive policy design - Real LLM integration."""

import asyncio
import os
from typing import Optional, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

router = APIRouter()


class ChatMessage(BaseModel):
    """Chat message."""
    role: str  # user, assistant, system
    content: str
    timestamp: Optional[str] = None


class ChatSession(BaseModel):
    """Chat session state."""
    id: str
    domain: str
    messages: List[ChatMessage] = []
    current_policy: Optional[dict] = None


# In-memory chat sessions
_sessions: dict[str, ChatSession] = {}

# System prompt for augmentation assistant
SYSTEM_PROMPT = """You are AugmentAI, an expert data augmentation advisor for computer vision tasks.

Your role is to help users design effective, domain-appropriate augmentation policies. You:

1. **Understand the domain**: Medical imaging, OCR, satellite imagery, etc. each have specific constraints.
2. **Recommend safe transforms**: Only suggest augmentations that are valid for the user's domain.
3. **Explain your reasoning**: Tell users WHY certain augmentations are good or bad for their task.
4. **Respect hard constraints**: NEVER suggest transforms that are FORBIDDEN in the domain.

DOMAIN CONSTRAINTS:
- **medical**: NO ElasticTransform, GridDistortion, ColorJitter (distorts anatomy)
- **ocr**: NO MotionBlur, ElasticTransform (destroys text legibility)
- **satellite**: NO ColorJitter, HueSaturationValue (alters spectral data)
- **natural**: All transforms allowed

When suggesting a policy, format it as YAML code block like:
```yaml
name: policy_name
domain: domain_type
transforms:
  - name: TransformName
    probability: 0.5
    parameters:
      param1: value1
```

Be concise but helpful. Ask clarifying questions when needed."""


@router.websocket("/ws")
async def chat_websocket(websocket: WebSocket):
    """WebSocket for interactive chat-based policy design."""
    await websocket.accept()
    
    import uuid
    session_id = str(uuid.uuid4())[:8]
    
    session = ChatSession(
        id=session_id,
        domain="natural",
        messages=[
            ChatMessage(
                role="assistant",
                content="👋 I'm your AugmentAI assistant! Tell me about your dataset and I'll help design an augmentation policy.\n\nFor example: *'I have chest X-ray images for pneumonia detection'* or *'I'm training an OCR model on receipts'*"
            )
        ],
    )
    _sessions[session_id] = session
    
    # Check if LLM is available
    llm_available = _check_llm_available()
    
    # Send initial state
    await websocket.send_json({
        "type": "init",
        "session_id": session_id,
        "messages": [m.model_dump() for m in session.messages],
        "llm_available": llm_available,
    })
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "message":
                user_message = data.get("content", "")
                
                # Add user message
                session.messages.append(ChatMessage(
                    role="user",
                    content=user_message,
                ))
                
                # Send typing indicator
                await websocket.send_json({"type": "typing", "typing": True})
                
                # Generate response
                response = await _generate_response(session, user_message)
                
                session.messages.append(ChatMessage(
                    role="assistant", 
                    content=response["content"],
                ))
                
                # Update policy if extracted
                if "policy" in response and response["policy"]:
                    session.current_policy = response["policy"]
                
                # Send response
                await websocket.send_json({
                    "type": "message",
                    "message": response,
                    "policy": session.current_policy,
                })
                
            elif data.get("type") == "set_domain":
                session.domain = data.get("domain", "natural")
                await websocket.send_json({
                    "type": "domain_changed",
                    "domain": session.domain,
                })
                
    except WebSocketDisconnect:
        if session_id in _sessions:
            del _sessions[session_id]


def _check_llm_available() -> bool:
    """Check if any LLM provider is configured."""
    return bool(
        os.getenv("OPENAI_API_KEY") or 
        os.getenv("GOOGLE_API_KEY") or
        os.getenv("GEMINI_API_KEY")
    )


async def _generate_response(session: ChatSession, user_input: str) -> dict:
    """Generate AI response (uses LLM if available, otherwise rule-based)."""
    
    # Try to use actual LLM
    if _check_llm_available():
        try:
            return await _llm_response(session, user_input)
        except Exception as e:
            # Log error and fallback
            print(f"LLM error: {e}")
    
    # Fallback to rule-based responses
    return _rule_based_response(session, user_input)


async def _llm_response(session: ChatSession, user_input: str) -> dict:
    """Generate response using actual LLM."""
    from augmentai.llm import LLMClient, Message
    from augmentai.llm.client import MessageRole
    
    # Build conversation history
    messages = [
        Message(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
    ]
    
    # Add conversation history (last 10 messages for context)
    for msg in session.messages[-10:]:
        role = MessageRole.USER if msg.role == "user" else MessageRole.ASSISTANT
        messages.append(Message(role=role, content=msg.content))
    
    # Add current user message
    messages.append(Message(role=MessageRole.USER, content=user_input))
    
    # Create client and get response
    client = LLMClient()
    
    # Run in executor to not block
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.chat(messages, temperature=0.7)
    )
    
    content = response.content
    
    # Try to extract policy from response
    policy = _extract_policy_from_response(content)
    
    # Update domain if detected
    if "medical" in user_input.lower() or "x-ray" in user_input.lower() or "mri" in user_input.lower():
        session.domain = "medical"
    elif "ocr" in user_input.lower() or "document" in user_input.lower() or "text" in user_input.lower():
        session.domain = "ocr"
    elif "satellite" in user_input.lower() or "aerial" in user_input.lower():
        session.domain = "satellite"
    
    return {
        "content": content,
        "domain": session.domain,
        "policy": policy,
    }


def _extract_policy_from_response(content: str) -> Optional[dict]:
    """Try to extract YAML policy from LLM response."""
    import yaml
    import re
    
    # Look for YAML code blocks
    yaml_match = re.search(r'```ya?ml\s*(.*?)```', content, re.DOTALL)
    if yaml_match:
        try:
            policy = yaml.safe_load(yaml_match.group(1))
            if isinstance(policy, dict) and "transforms" in policy:
                return policy
        except:
            pass
    
    return None


def _rule_based_response(session: ChatSession, user_input: str) -> dict:
    """Generate rule-based response for demo (when no LLM available)."""
    user_lower = user_input.lower()
    
    # Detect intent
    if any(word in user_lower for word in ["medical", "xray", "x-ray", "ct", "mri", "radiology", "chest"]):
        session.domain = "medical"
        session.current_policy = {
            "name": "medical_policy",
            "domain": "medical",
            "transforms": [
                {"name": "HorizontalFlip", "probability": 0.5},
                {"name": "Rotate", "probability": 0.3, "parameters": {"limit": 10}},
                {"name": "CLAHE", "probability": 0.5},
                {"name": "GaussNoise", "probability": 0.2},
            ]
        }
        return {
            "content": f"I've set the domain to **medical imaging**. For medical data, I've created a safe policy that avoids geometric distortions that could alter anatomical structures.\n\n```yaml\nname: medical_policy\ndomain: medical\ntransforms:\n  - name: HorizontalFlip\n    probability: 0.5\n  - name: Rotate\n    probability: 0.3\n    parameters:\n      limit: 10\n  - name: CLAHE\n    probability: 0.5\n  - name: GaussNoise\n    probability: 0.2\n```\n\n**Why these?**\n- ✅ HorizontalFlip: Safe for most anatomical structures\n- ✅ Rotate ±10°: Mild rotation preserves anatomy\n- ✅ CLAHE: Enhances contrast without distorting\n- ✅ GaussNoise: Simulates sensor noise\n\n❌ **Avoided**: ElasticTransform, GridDistortion (could create fake pathology)\n\nWould you like to adjust any parameters?",
            "domain": "medical",
            "policy": session.current_policy,
        }
    
    elif any(word in user_lower for word in ["ocr", "document", "text", "receipt"]):
        session.domain = "ocr"
        session.current_policy = {
            "name": "ocr_policy",
            "domain": "ocr",
            "transforms": [
                {"name": "Rotate", "probability": 0.3, "parameters": {"limit": 5}},
                {"name": "RandomBrightnessContrast", "probability": 0.4},
            ]
        }
        return {
            "content": "I've set the domain to **OCR/documents**. For text recognition, I avoid transforms that could blur or distort text.\n\n```yaml\nname: ocr_policy\ndomain: ocr\ntransforms:\n  - name: Rotate\n    probability: 0.3\n    parameters:\n      limit: 5\n  - name: RandomBrightnessContrast\n    probability: 0.4\n```\n\n**Why these?**\n- ✅ Slight rotation (±5°): Simulates scanning angle variance\n- ✅ Brightness/Contrast: Handles different paper/lighting\n\n❌ **Avoided**: MotionBlur, ElasticTransform (destroys text legibility)",
            "domain": "ocr",
            "policy": session.current_policy,
        }
    
    elif any(word in user_lower for word in ["satellite", "aerial", "remote sensing"]):
        session.domain = "satellite"
        session.current_policy = {
            "name": "satellite_policy",
            "domain": "satellite",
            "transforms": [
                {"name": "HorizontalFlip", "probability": 0.5},
                {"name": "VerticalFlip", "probability": 0.5},
                {"name": "Rotate", "probability": 0.5, "parameters": {"limit": 180}},
            ]
        }
        return {
            "content": "I've set the domain to **satellite/remote sensing**. For aerial imagery, I avoid color transforms that could alter spectral relationships.\n\n```yaml\nname: satellite_policy\ndomain: satellite\ntransforms:\n  - name: HorizontalFlip\n    probability: 0.5\n  - name: VerticalFlip\n    probability: 0.5\n  - name: Rotate\n    probability: 0.5\n    parameters:\n      limit: 180\n```\n\n**Why these?**\n- ✅ All flips/rotations: Satellite images have no canonical orientation\n- ✅ Full 180° rotation: Captures all viewing angles\n\n❌ **Avoided**: ColorJitter, HueSaturationValue (alters spectral data)",
            "domain": "satellite",
            "policy": session.current_policy,
        }
    
    elif any(word in user_lower for word in ["export", "save", "download"]):
        if session.current_policy:
            return {
                "content": "You can export the current policy using:\n\n**CLI:**\n```bash\naugmentai export policy.yaml --format python\n```\n\n**Or copy the YAML from above and save it to a file!**\n\nWould you like me to show it in a different format?",
                "policy": session.current_policy,
            }
        else:
            return {
                "content": "No policy created yet! Tell me about your dataset first. For example:\n\n- *'I have chest X-ray images'*\n- *'I'm training OCR on receipts'*\n- *'I have satellite imagery for land classification'*",
            }
    
    else:
        # Default response
        session.current_policy = {
            "name": "default_policy",
            "domain": session.domain,
            "transforms": [
                {"name": "HorizontalFlip", "probability": 0.5},
                {"name": "Rotate", "probability": 0.3, "parameters": {"limit": 15}},
                {"name": "RandomBrightnessContrast", "probability": 0.4},
                {"name": "GaussNoise", "probability": 0.2},
            ]
        }
        return {
            "content": f"I've created a general-purpose policy for **{session.domain}** images:\n\n```yaml\nname: default_policy\ndomain: {session.domain}\ntransforms:\n  - name: HorizontalFlip\n    probability: 0.5\n  - name: Rotate\n    probability: 0.3\n    parameters:\n      limit: 15\n  - name: RandomBrightnessContrast\n    probability: 0.4\n  - name: GaussNoise\n    probability: 0.2\n```\n\nTell me more about your use case for domain-specific recommendations:\n- 🏥 **Medical** (X-ray, CT, MRI)\n- 📄 **OCR** (documents, receipts)\n- 🛰️ **Satellite** (aerial imagery)\n- 📷 **Natural** (general photos)",
            "policy": session.current_policy,
        }
