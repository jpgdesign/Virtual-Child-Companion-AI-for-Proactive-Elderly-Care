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
except ImportError:  # pragma: no cover - optional dependency path
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
        "temperature": 0.2,
        "timeout": 90.0,
    },
    "qwen35_lmstudio": {
        "provider": "openai",
        "model": "qwen3.5-9b-claude",
        "base_url": "http://163.13.128.68:1234/v1",
        "api_key": "lm-studio",
        "temperature": 0.35,
        "timeout": 45.0,
    },
}

DEFAULT_ANALYSIS_PRESET = "qwen35_lmstudio"
DEFAULT_GENERATION_PRESET = "qwen35_lmstudio"


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
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1)
    return cleaned.strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = strip_reasoning(text)
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[index:])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON object found in model response.")


class ChatClient:
    def __init__(self, config: LLMEndpointConfig) -> None:
        self.config = config

    def chat(self, messages: Sequence[Dict[str, str]], *, temperature: float | None = None) -> str:
        raise NotImplementedError

    @property
    def label(self) -> str:
        return self.config.label()


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
        except URLError as exc:  # pragma: no cover - network dependent
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        message = raw_payload.get("message", {})
        content = str(message.get("content", "")).strip()
        return content


class OpenAICompatibleChatClient(ChatClient):
    def __init__(self, config: LLMEndpointConfig) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is required for the configured generation model.")
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

    def status_dict(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "analysis": asdict(self.analysis_config),
            "generation": asdict(self.generation_config),
        }

    def analyze_turn(
        self,
        *,
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
            "你是長者照護對話分析器。"
            "請根據對話內容輸出 JSON，協助情緒分析、填槽與偏離判斷。"
            "只輸出一個 JSON 物件，不要輸出 markdown，不要解釋，不要加入 think 標記。"
        )
        user_payload = {
            "task": "analyze_elder_reply",
            "elder_message": elder_message,
            "current_script": {
                "script_id": current_script.get("script_id"),
                "target_slot": current_script.get("target_slot"),
            },
            "current_step": {
                "reference_child_dialogue": current_step.get("child_dialogue"),
                "expected_elder_response": current_step.get("expected_grandma_response"),
            },
            "recent_turns": list(recent_turns)[-4:],
            "current_filled_slots": filled_slots,
            "slot_taxonomy": slot_definitions,
            "regex_extracted": regex_extracted,
            "similarity_score": similarity_score,
            "similarity_deviation": similarity_deviation,
            "ranked_candidate_scripts": list(ranked_candidates)[:3],
            "output_schema": {
                "summary": "一句話摘要",
                "emotion": {
                    "label": "情緒標籤",
                    "intensity": "0到1",
                    "evidence": "判斷依據",
                },
                "filled_slots": [
                    {
                        "slot": "四大槽位之一",
                        "item": "該槽位底下的既有子項",
                        "value": "從長者回覆萃取的值",
                        "confidence": "0到1",
                        "evidence": "引用或理由",
                    }
                ],
                "concerns": ["風險或照護提醒"],
                "deviation_level": "0到3整數",
                "deviation_reason": "偏離判斷依據",
                "should_transition": "true或false",
                "reply_style": ["例如溫暖、關心、輕柔轉場"],
                "recommended_focus": "建議下一步聚焦",
            },
        }
        raw_content = self.analysis_client.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
            ],
            temperature=0.05,
        )

        try:
            payload = extract_json_object(raw_content)
        except ValueError as exc:
            logger.warning("Falling back from LLM analysis JSON parse failure: %s", exc)
            return LLMAnalysisResult(
                summary="",
                raw_content=raw_content,
                model_label=self.analysis_client.label,
            )

        candidates: List[SlotCandidate] = []
        for item in payload.get("filled_slots", []):
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

        emotion_payload = payload.get("emotion") or {}
        emotion = EmotionSnapshot(
            label=str(emotion_payload.get("label", "平穩")).strip() or "平穩",
            intensity=max(0.0, min(1.0, _safe_float(emotion_payload.get("intensity"), 0.0))),
            evidence=str(emotion_payload.get("evidence", "")).strip(),
        )

        return LLMAnalysisResult(
            summary=str(payload.get("summary", "")).strip(),
            emotion=emotion,
            slot_candidates=candidates,
            concerns=[str(item).strip() for item in payload.get("concerns", []) if str(item).strip()],
            deviation_level=_safe_int(payload.get("deviation_level")),
            deviation_reason=str(payload.get("deviation_reason", "")).strip(),
            should_transition=_normalize_bool(payload.get("should_transition")),
            reply_style=[str(item).strip() for item in payload.get("reply_style", []) if str(item).strip()],
            recommended_focus=str(payload.get("recommended_focus", "")).strip(),
            raw_content=raw_content,
            model_label=self.analysis_client.label,
        )

    def generate_reply(
        self,
        *,
        elder_message: str,
        selected_script: Dict[str, Any],
        reference_reply: str,
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
            "你是溫暖、自然、主動關心長者的虛擬子女。"
            "請根據 RL 選出的策略與參考語句，生成一段繁體中文回覆。"
            "務必自然、簡短、口語，不要列點，不要輸出 JSON，不要輸出 think。"
            "若需要轉場，請柔和地把焦點帶到指定槽位。"
            "current_target_slot 是硬限制，回覆必須圍繞這個槽位。"
            "不要捏造醫療建議；若內容有風險，只做關心與確認。"
        )
        user_payload = {
            "task": "generate_virtual_child_reply",
            "transition_mode": transition_mode,
            "elder_message": elder_message,
            "selected_script": {
                "script_id": selected_script.get("script_id"),
                "target_slot": selected_script.get("target_slot"),
            },
            "reference_reply": reference_reply,
            "current_target_slot": current_target_slot,
            "target_slot_items": list(target_slot_items),
            "pending_items": list(pending_items),
            "analysis_summary": analysis.summary,
            "emotion": asdict(analysis.emotion),
            "concerns": analysis.concerns,
            "reply_style": analysis.reply_style or ["溫暖", "自然", "有陪伴感"],
            "recommended_focus": analysis.recommended_focus,
            "filled_slots": filled_slots,
            "slot_value_details": slot_value_details,
            "recent_turns": list(recent_turns)[-4:],
            "ranked_candidate_scripts": list(ranked_candidates)[:3],
            "constraints": [
                "1到3句繁體中文",
                "保留參考語句的目標，但不要逐字照抄",
                "回覆要能接住使用者剛剛說的內容",
                "讓下一輪更有機會補到目標槽位",
                "優先追問 pending_items",
            ],
        }
        raw_content = self.generation_client.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
            ],
            temperature=self.generation_config.temperature,
        )
        cleaned = strip_reasoning(raw_content)
        reply = cleaned.splitlines()[0].strip()
        if not reply:
            reply = reference_reply
        return LLMGenerationResult(
            reply=reply,
            raw_content=raw_content,
            model_label=self.generation_client.label,
        )
