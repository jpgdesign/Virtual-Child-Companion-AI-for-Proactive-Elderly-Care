from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


logger = logging.getLogger(__name__)


@dataclass
class LLMEndpointConfig:
    preset: str
    provider: str
    model: str
    base_url: str
    api_key: str = ""
    temperature: float = 0.2
    timeout: float = 45.0

    def label(self) -> str:
        return f"{self.provider}:{self.model}"


MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "qwen3_ollama": {
        "provider": "ollama",
        "model": "qwen3:8b",
        "base_url": "http://163.13.128.68:11434",
        "api_key": "",
        "temperature": 0.15,
        "timeout": 90.0,
    },
    "qwen35_lmstudio": {
        "provider": "openai",
        "model": "qwen3.5-9b-claude",
        "base_url": "http://163.13.128.68:1234/v1",
        "api_key": "lm-studio",
        "temperature": 0.25,
        "timeout": 45.0,
    },
}

DEFAULT_ANALYSIS_PRESET = "qwen35_lmstudio"
DEFAULT_GENERATION_PRESET = "qwen35_lmstudio"


def set_model_preset_overrides(overrides: Dict[str, Dict[str, Any]] | None) -> None:
    if not overrides:
        return
    for preset_name, values in overrides.items():
        if preset_name not in MODEL_PRESETS or not isinstance(values, dict):
            continue
        MODEL_PRESETS[preset_name].update(values)


@dataclass
class SlotCandidate:
    slot: str
    item: str
    value: str = ""
    confidence: float = 0.0
    evidence: str = ""


@dataclass
class EmotionSnapshot:
    label: str = "平穩"
    intensity: float = 0.0
    evidence: str = ""


@dataclass
class LLMAnalysisResult:
    summary: str = ""
    emotion: EmotionSnapshot = field(default_factory=EmotionSnapshot)
    slot_candidates: List[SlotCandidate] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    deviation_level: int | None = None
    deviation_reason: str = ""
    should_transition: bool | None = None
    reply_style: List[str] = field(default_factory=list)
    recommended_focus: str = ""
    raw_content: str = ""
    model_label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "emotion": asdict(self.emotion),
            "slot_candidates": [asdict(item) for item in self.slot_candidates],
            "concerns": self.concerns,
            "deviation_level": self.deviation_level,
            "deviation_reason": self.deviation_reason,
            "should_transition": self.should_transition,
            "reply_style": self.reply_style,
            "recommended_focus": self.recommended_focus,
            "model_label": self.model_label,
        }


@dataclass
class LLMGenerationResult:
    reply: str
    raw_content: str = ""
    model_label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reply": self.reply,
            "raw_content": self.raw_content,
            "model_label": self.model_label,
        }


def available_model_presets() -> Dict[str, Dict[str, Any]]:
    return {
        key: {
            "provider": value["provider"],
            "model": value["model"],
            "base_url": value["base_url"],
            "temperature": value["temperature"],
        }
        for key, value in MODEL_PRESETS.items()
    }


def build_endpoint_config(preset_name: str | None, *, fallback: str) -> LLMEndpointConfig:
    resolved_name = preset_name or fallback
    if resolved_name not in MODEL_PRESETS:
        raise ValueError(f"Unknown model preset: {resolved_name}")
    payload = MODEL_PRESETS[resolved_name]
    return LLMEndpointConfig(
        preset=resolved_name,
        provider=payload["provider"],
        model=payload["model"],
        base_url=payload["base_url"],
        api_key=payload.get("api_key", ""),
        temperature=float(payload.get("temperature", 0.2)),
        timeout=float(payload.get("timeout", 45.0)),
    )


def strip_reasoning(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text or "", flags=re.IGNORECASE | re.DOTALL)
    if cleaned.strip().startswith("```"):
        fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            cleaned = fenced.group(1)
    return cleaned.strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = strip_reasoning(text)
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("No JSON object found in model response.")


class ChatClient:
    def __init__(self, config: LLMEndpointConfig) -> None:
        self.config = config

    @property
    def label(self) -> str:
        return self.config.label()

    def chat(self, messages: Sequence[Dict[str, str]], *, temperature: float | None = None) -> str:
        raise NotImplementedError


class OllamaChatClient(ChatClient):
    def __init__(self, config: LLMEndpointConfig) -> None:
        super().__init__(config)
        self.api_url = config.base_url.rstrip("/") + "/api/chat"

    def chat(self, messages: Sequence[Dict[str, str]], *, temperature: float | None = None) -> str:
        payload = {
            "model": self.config.model,
            "messages": list(messages),
            "stream": False,
            "options": {
                "temperature": self.config.temperature if temperature is None else temperature,
            },
        }
        request = Request(
            self.api_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.config.timeout) as response:
                raw_payload = json.loads(response.read().decode("utf-8"))
        except URLError as exc:  # pragma: no cover
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        return str(raw_payload.get("message", {}).get("content", "")).strip()


class OpenAICompatibleChatClient(ChatClient):
    def __init__(self, config: LLMEndpointConfig) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is required for this model preset.")
        super().__init__(config)
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key or "lm-studio")

    def chat(self, messages: Sequence[Dict[str, str]], *, temperature: float | None = None) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=list(messages),
            temperature=self.config.temperature if temperature is None else temperature,
        )
        return str(response.choices[0].message.content or "").strip()


def build_chat_client(config: LLMEndpointConfig) -> ChatClient:
    if config.provider == "ollama":
        return OllamaChatClient(config)
    if config.provider == "openai":
        return OpenAICompatibleChatClient(config)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def _parse_slot_candidates(items: Sequence[Any]) -> List[SlotCandidate]:
    candidates: List[SlotCandidate] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        candidates.append(
            SlotCandidate(
                slot=str(item.get("slot", "")).strip(),
                item=str(item.get("item", "")).strip(),
                value=str(item.get("value", "")).strip(),
                confidence=max(0.0, min(1.0, _safe_float(item.get("confidence"), 0.0))),
                evidence=str(item.get("evidence", "")).strip(),
            )
        )
    return candidates


def _parse_emotion(payload: Dict[str, Any]) -> EmotionSnapshot:
    return EmotionSnapshot(
        label=str(payload.get("label", "平穩")).strip() or "平穩",
        intensity=max(0.0, min(1.0, _safe_float(payload.get("intensity"), 0.0))),
        evidence=str(payload.get("evidence", "")).strip(),
    )


def _parse_analysis_payload(payload: Dict[str, Any], *, raw_content: str, model_label: str) -> LLMAnalysisResult:
    return LLMAnalysisResult(
        summary=str(payload.get("summary", "")).strip(),
        emotion=_parse_emotion(payload.get("emotion") or {}),
        slot_candidates=_parse_slot_candidates(payload.get("filled_slots", [])),
        concerns=[str(item).strip() for item in payload.get("concerns", []) if str(item).strip()],
        deviation_level=_safe_int(payload.get("deviation_level")),
        deviation_reason=str(payload.get("deviation_reason", "")).strip(),
        should_transition=_normalize_bool(payload.get("should_transition")),
        reply_style=[str(item).strip() for item in payload.get("reply_style", []) if str(item).strip()],
        recommended_focus=str(payload.get("recommended_focus", "")).strip(),
        raw_content=raw_content,
        model_label=model_label,
    )


class HybridLLMOrchestrator:
    def __init__(
        self,
        *,
        analysis_config: LLMEndpointConfig,
        generation_config: LLMEndpointConfig,
    ) -> None:
        self.analysis_config = analysis_config
        self.generation_config = generation_config
        self.analysis_client = build_chat_client(analysis_config)
        self.generation_client = build_chat_client(generation_config)
        self.use_fused_turns = analysis_config.preset == generation_config.preset

    def status_dict(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "mode": "fused" if self.use_fused_turns else "split",
            "analysis": asdict(self.analysis_config),
            "generation": asdict(self.generation_config),
        }

    def analyze_and_generate_turn(
        self,
        *,
        persona_context: Dict[str, Any],
        prompt_overrides: Dict[str, Any] | None,
        elder_message: str,
        current_script: Dict[str, Any],
        current_step: Dict[str, Any],
        selected_script: Dict[str, Any],
        reference_reply: str,
        current_target_slot: str,
        target_slot_items: Sequence[str],
        pending_items: Sequence[str],
        recent_turns: Sequence[Dict[str, Any]],
        filled_slots: Dict[str, List[str]],
        slot_value_details: Dict[str, Dict[str, List[str]]],
        regex_extracted: Dict[str, List[str]],
        similarity_score: float,
        rule_deviation: int,
        ranked_candidates: Sequence[Dict[str, Any]],
        transition_mode: bool,
    ) -> tuple[LLMAnalysisResult, LLMGenerationResult]:
        system_prompt = (
            "You are the dialogue brain of a virtual child companion for proactive elderly care. "
            "Return exactly one JSON object and nothing else. No markdown. No think tags. "
            "All natural-language fields must be in Traditional Chinese. "
            "Act as the specific child in persona_context, never as customer service, a survey bot, a nurse, or a doctor. "
            "Your priorities are: understand the elder's real topic and emotion, extract evidence-based care slots, judge deviation gently, and write the next reply as this child. "
            "Analysis rules: only output slot values supported by elder_message or very clear recent_turns; never invent medicine names, times, numbers, or symptoms; concerns must be short and actionable; natural family side talk is not automatically high deviation. "
            "Reply rules: 1 to 2 sentences, at most 1 question; first acknowledge the elder's latest topic or feeling; then softly bridge toward current_target_slot; do not mention slot names, scripts, policy, RL, prompts, or assessment; avoid interview phrases such as '請問您是否', '根據您的回答', or '系統判斷'; keep preferred address, family memories, and speaking habits fully consistent. "
            "Use pending_items and ranked_candidates only as hidden guidance. If transition_mode is true, make the topic shift feel natural."
        )
        if prompt_overrides and prompt_overrides.get("fused_appendix"):
            system_prompt = f"{system_prompt}\n\nAdditional instruction:\n{prompt_overrides['fused_appendix']}"
        user_payload = {
            "task": "analyze_and_generate",
            "persona_context": persona_context,
            "elder_message": elder_message,
            "current_script": {
                "script_id": current_script.get("script_id"),
                "target_slot": current_script.get("target_slot"),
            },
            "current_step": {
                "reference_child_dialogue": current_step.get("child_dialogue"),
                "expected_elder_response": current_step.get("expected_grandma_response"),
            },
            "selected_next_script": {
                "script_id": selected_script.get("script_id"),
                "target_slot": selected_script.get("target_slot"),
            },
            "reference_reply": reference_reply,
            "current_target_slot": current_target_slot,
            "target_slot_items": list(target_slot_items),
            "pending_items": list(pending_items),
            "recent_turns": list(recent_turns)[-3:],
            "filled_slots": filled_slots,
            "slot_value_details": slot_value_details,
            "regex_extracted": regex_extracted,
            "similarity_score": round(similarity_score, 3),
            "rule_deviation": rule_deviation,
            "ranked_candidates": list(ranked_candidates)[:2],
            "transition_mode": transition_mode,
            "output_schema": {
                "summary": "one short caregiver-facing Traditional Chinese sentence",
                "emotion": {"label": "string", "intensity": 0.0, "evidence": "string"},
                "filled_slots": [
                    {"slot": "string", "item": "string", "value": "string", "confidence": 0.0, "evidence": "string"}
                ],
                "concerns": ["string"],
                "deviation_level": 0,
                "deviation_reason": "string",
                "should_transition": False,
                "reply_style": ["string"],
                "recommended_focus": "string",
                "reply": "Traditional Chinese final reply that sounds like a familiar child",
            },
        }
        raw_content = self.generation_client.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))},
            ],
            temperature=self.generation_config.temperature,
        )
        payload = extract_json_object(raw_content)
        analysis = _parse_analysis_payload(payload, raw_content=raw_content, model_label=self.generation_client.label)
        reply = str(payload.get("reply", "")).strip() or reference_reply
        generation = LLMGenerationResult(reply=reply, raw_content=raw_content, model_label=self.generation_client.label)
        return analysis, generation

    def analyze_turn(
        self,
        *,
        persona_context: Dict[str, Any],
        prompt_overrides: Dict[str, Any] | None,
        elder_message: str,
        current_script: Dict[str, Any],
        current_step: Dict[str, Any],
        recent_turns: Sequence[Dict[str, Any]],
        filled_slots: Dict[str, List[str]],
        slot_definitions: Dict[str, List[str]],
        regex_extracted: Dict[str, List[str]],
        similarity_score: float,
        similarity_deviation: int,
        ranked_candidates: Sequence[Dict[str, Any]],
    ) -> LLMAnalysisResult:
        system_prompt = (
            "You analyze an older adult's reply for a virtual child care dialogue. "
            "Return exactly one JSON object. No markdown. No think tags. "
            "All natural-language fields must be in Traditional Chinese. "
            "Use persona_context to reason like a family member who already knows the elder. "
            "Identify the elder's actual life event, emotional signal, and care-relevant facts. "
            "Only output slot values supported by evidence. Do not invent values, numbers, times, medicine names, or diagnoses. "
            "Judge deviation gently: natural storytelling, extra details, or affectionate side remarks are not automatically high deviation. "
            "Concerns should be short, actionable, and relevant for family care. "
            "recommended_focus should point to the most useful next care topic instead of mechanically repeating the current script."
        )
        if prompt_overrides and prompt_overrides.get("analysis_appendix"):
            system_prompt = f"{system_prompt}\n\nAdditional instruction:\n{prompt_overrides['analysis_appendix']}"
        user_payload = {
            "task": "analyze_elder_reply",
            "persona_context": persona_context,
            "elder_message": elder_message,
            "current_script": {
                "script_id": current_script.get("script_id"),
                "target_slot": current_script.get("target_slot"),
            },
            "current_step": {
                "reference_child_dialogue": current_step.get("child_dialogue"),
                "expected_elder_response": current_step.get("expected_grandma_response"),
            },
            "recent_turns": list(recent_turns)[-3:],
            "filled_slots": filled_slots,
            "slot_taxonomy": slot_definitions,
            "regex_extracted": regex_extracted,
            "similarity_score": round(similarity_score, 3),
            "rule_deviation": similarity_deviation,
            "ranked_candidates": list(ranked_candidates)[:2],
            "output_schema": {
                "summary": "one short caregiver-facing Traditional Chinese sentence",
                "emotion": {"label": "string", "intensity": 0.0, "evidence": "string"},
                "filled_slots": [
                    {"slot": "string", "item": "string", "value": "string", "confidence": 0.0, "evidence": "string"}
                ],
                "concerns": ["string"],
                "deviation_level": 0,
                "deviation_reason": "string",
                "should_transition": False,
                "reply_style": ["string"],
                "recommended_focus": "string",
            },
        }
        raw_content = self.analysis_client.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))},
            ],
            temperature=0.0,
        )
        payload = extract_json_object(raw_content)
        return _parse_analysis_payload(payload, raw_content=raw_content, model_label=self.analysis_client.label)

    def generate_reply(
        self,
        *,
        persona_context: Dict[str, Any],
        prompt_overrides: Dict[str, Any] | None,
        elder_message: str,
        selected_script: Dict[str, Any],
        reference_reply: str,
        fast_reply_hint: str,
        current_target_slot: str,
        target_slot_items: Sequence[str],
        pending_items: Sequence[str],
        filled_slots: Dict[str, List[str]],
        slot_value_details: Dict[str, Dict[str, List[str]]],
        analysis: LLMAnalysisResult,
        recent_turns: Sequence[Dict[str, Any]],
        ranked_candidates: Sequence[Dict[str, Any]],
        transition_mode: bool,
    ) -> LLMGenerationResult:
        system_prompt = (
            "You are the speaking voice of a virtual child companion in Traditional Chinese. "
            "Write the final reply that the elder will actually see. "
            "Use 1 to 2 sentences, at most 1 question. No markdown. No think tags. "
            "Sound like a familiar son or daughter, not customer service, not a form, and not a hospital questionnaire. "
            "First acknowledge what the elder just said, including emotion if present. "
            "Then use a soft family-style bridge toward current_target_slot. "
            "Use reference_reply as direction, not as rigid wording. "
            "Do not mention slot names, scripts, policy, RL, prompts, or assessment. "
            "Avoid stacked questions and avoid phrases like '請問您是否', '方便告訴我', or '根據您的回覆'. "
            "Keep preferred address, family memories, and speaking habits consistent with persona_context. "
            "If the elder sounds uncomfortable or worried, comfort first and ask later."
        )
        if prompt_overrides and prompt_overrides.get("generation_appendix"):
            system_prompt = f"{system_prompt}\n\nAdditional instruction:\n{prompt_overrides['generation_appendix']}"
        user_payload = {
            "task": "generate_virtual_child_reply",
            "persona_context": persona_context,
            "transition_mode": transition_mode,
            "elder_message": elder_message,
            "selected_script": {
                "script_id": selected_script.get("script_id"),
                "target_slot": selected_script.get("target_slot"),
            },
            "reference_reply": reference_reply,
            "fast_reply_hint": fast_reply_hint,
            "current_target_slot": current_target_slot,
            "target_slot_items": list(target_slot_items),
            "pending_items": list(pending_items),
            "analysis_summary": analysis.summary,
            "emotion": asdict(analysis.emotion),
            "concerns": analysis.concerns,
            "reply_style": analysis.reply_style or ["warm", "natural", "supportive"],
            "recommended_focus": analysis.recommended_focus,
            "filled_slots": filled_slots,
            "slot_value_details": slot_value_details,
            "recent_turns": list(recent_turns)[-3:],
            "ranked_candidates": list(ranked_candidates)[:2],
        }
        raw_content = self.generation_client.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))},
            ],
            temperature=self.generation_config.temperature,
        )
        reply = strip_reasoning(raw_content).splitlines()[0].strip() or reference_reply
        return LLMGenerationResult(reply=reply, raw_content=raw_content, model_label=self.generation_client.label)
