from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from llm_runtime import (
    DEFAULT_ANALYSIS_PRESET,
    DEFAULT_GENERATION_PRESET,
    MODEL_PRESETS,
    set_model_preset_overrides,
)
from persona_profiles import DEFAULT_PERSONA_PROFILE_ID, PERSONA_PROFILES


STATE_PATH = Path("artifacts/platform_state.json")


LEGACY_PROMPT_SETTINGS = {
    "analysis_appendix": "分析時要特別參考家庭關係畫像、長者個性與既有相處模式，不要只做一般客服式判讀。",
    "generation_appendix": "回覆必須像熟悉長者的兒女，不要像客服或醫院問卷。要先接話，再隱性引導。",
    "fused_appendix": "請保持兒女人設一致，優先延續家人原本的講話方式、稱呼與共同回憶。",
}


DEFAULT_PROMPT_SETTINGS = {
    "analysis_appendix": (
        "請優先理解長者此刻真正想表達的生活事件、情緒和照護線索。"
        "填槽必須有語意依據，不能憑空補出數值、藥名、時間或症狀。"
        "如果只是自然延伸聊天，不要過度判成高偏離。"
        "concerns 只保留家屬真的需要留意的重點，summary 要寫成一看就懂的繁中短句。"
    ),
    "generation_appendix": (
        "你不是客服、護理師或問卷系統，而是熟悉長者的兒女。"
        "回覆要先接住對方剛說的內容與情緒，再用一句自然過橋慢慢帶到目標槽位。"
        "避免連珠炮提問、避免教條式關心、避免直接說槽位名稱。"
        "以 1 到 2 句為主，最多 1 個問題，語氣要像家人。"
    ),
    "fused_appendix": (
        "請同時完成分析與生成，但以家庭感、自然感和連續性為最高優先。"
        "若長者提到不舒服、擔心、疲累、睡不好、忘記、沒胃口，先安撫再引導。"
        "若已有足夠資料，不要硬追問；若缺資料，就挑最自然的一個缺口輕輕問。"
        "所有自然語句請用繁體中文，維持同一 persona、稱呼、共同回憶與說話習慣。"
    ),
}


DEFAULT_API_SETTINGS = {
    "default_algorithm": "dqn",
    "default_analysis_preset": DEFAULT_ANALYSIS_PRESET,
    "default_generation_preset": DEFAULT_GENERATION_PRESET,
    "model_overrides": deepcopy(MODEL_PRESETS),
}


DEFAULT_USERS: List[Dict[str, Any]] = [
    {
        "id": "u-family-01",
        "username": "xiaowen.family",
        "password": "XWFamily#2026",
        "display_name": "曉雯家屬帳號",
        "role": "family",
        "persona_profile_id": "daughter_teacher_mother",
        "default_view": "family",
        "enabled": True,
    },
    {
        "id": "u-elder-01",
        "username": "yulan.elder",
        "password": "YuLan#2026",
        "display_name": "玉蘭長者帳號",
        "role": "elder",
        "persona_profile_id": "daughter_teacher_mother",
        "default_view": "elder",
        "enabled": True,
    },
    {
        "id": "u-family-02",
        "username": "jiahao.family",
        "password": "JHFamily#2026",
        "display_name": "家豪家屬帳號",
        "role": "family",
        "persona_profile_id": "son_engineer_father",
        "default_view": "family",
        "enabled": True,
    },
    {
        "id": "u-elder-02",
        "username": "zhengxiong.elder",
        "password": "ZXElder#2026",
        "display_name": "正雄長者帳號",
        "role": "elder",
        "persona_profile_id": "son_engineer_father",
        "default_view": "elder",
        "enabled": True,
    },
    {
        "id": "u-family-03",
        "username": "yating.family",
        "password": "YTFamily#2026",
        "display_name": "雅婷家屬帳號",
        "role": "family",
        "persona_profile_id": "daughter_nurse_mother",
        "default_view": "family",
        "enabled": True,
    },
    {
        "id": "u-elder-03",
        "username": "xiuqin.elder",
        "password": "XiuQin#2026",
        "display_name": "秀琴長者帳號",
        "role": "elder",
        "persona_profile_id": "daughter_nurse_mother",
        "default_view": "elder",
        "enabled": True,
    },
    {
        "id": "u-admin-01",
        "username": "admin.console",
        "password": "Admin#2026",
        "display_name": "平台管理員",
        "role": "admin",
        "persona_profile_id": DEFAULT_PERSONA_PROFILE_ID,
        "default_view": "admin",
        "enabled": True,
    },
]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def build_default_state() -> Dict[str, Any]:
    return {
        "users": deepcopy(DEFAULT_USERS),
        "personas": deepcopy(PERSONA_PROFILES),
        "prompt_settings": deepcopy(DEFAULT_PROMPT_SETTINGS),
        "api_settings": deepcopy(DEFAULT_API_SETTINGS),
        "conversation_records": [],
        "updated_at": _now_iso(),
    }


def _merge_prompt_settings(saved_prompt_settings: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(saved_prompt_settings, dict):
        return deepcopy(DEFAULT_PROMPT_SETTINGS)
    if saved_prompt_settings == LEGACY_PROMPT_SETTINGS:
        return deepcopy(DEFAULT_PROMPT_SETTINGS)
    merged = deepcopy(DEFAULT_PROMPT_SETTINGS)
    merged.update({key: str(value) for key, value in saved_prompt_settings.items()})
    return merged


def _merge_defaults(state: Dict[str, Any]) -> Dict[str, Any]:
    defaults = build_default_state()
    merged = defaults
    merged.update(state or {})
    merged["users"] = state.get("users", defaults["users"]) if isinstance(state, dict) else defaults["users"]
    merged["personas"] = state.get("personas", defaults["personas"]) if isinstance(state, dict) else defaults["personas"]
    merged["prompt_settings"] = _merge_prompt_settings(state.get("prompt_settings")) if isinstance(state, dict) else deepcopy(defaults["prompt_settings"])
    merged["api_settings"] = state.get("api_settings", defaults["api_settings"]) if isinstance(state, dict) else defaults["api_settings"]
    merged["conversation_records"] = state.get("conversation_records", defaults["conversation_records"]) if isinstance(state, dict) else defaults["conversation_records"]
    return merged


def load_platform_state() -> Dict[str, Any]:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not STATE_PATH.exists():
        state = build_default_state()
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        set_model_preset_overrides(state["api_settings"].get("model_overrides"))
        return state

    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    merged = _merge_defaults(state)
    set_model_preset_overrides(merged["api_settings"].get("model_overrides"))
    return merged


def save_platform_state(state: Dict[str, Any]) -> Dict[str, Any]:
    state["updated_at"] = _now_iso()
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    set_model_preset_overrides(state.get("api_settings", {}).get("model_overrides"))
    return state


def sanitize_user(user: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": user["id"],
        "username": user["username"],
        "display_name": user.get("display_name", user["username"]),
        "role": user.get("role", "family"),
        "persona_profile_id": user.get("persona_profile_id", DEFAULT_PERSONA_PROFILE_ID),
        "default_view": user.get("default_view", "family"),
        "enabled": bool(user.get("enabled", True)),
    }


def list_login_accounts(state: Dict[str, Any], *, include_admin: bool = False, include_password: bool = True) -> List[Dict[str, Any]]:
    accounts: List[Dict[str, Any]] = []
    for user in state.get("users", []):
        if not include_admin and user.get("role") == "admin":
            continue
        record = sanitize_user(user)
        if include_password:
            record["password"] = user.get("password", "")
        persona = state.get("personas", {}).get(record["persona_profile_id"], {})
        record["persona_label"] = persona.get("label", record["persona_profile_id"])
        accounts.append(record)
    return accounts


def authenticate_user(state: Dict[str, Any], username: str, password: str, *, require_admin: bool = False) -> Dict[str, Any] | None:
    for user in state.get("users", []):
        if not user.get("enabled", True):
            continue
        if user.get("username") != username or user.get("password") != password:
            continue
        if require_admin and user.get("role") != "admin":
            return None
        return sanitize_user(user)
    return None


def get_persona_profile(state: Dict[str, Any], persona_profile_id: str) -> Dict[str, Any]:
    personas = state.get("personas", {})
    profile = personas.get(persona_profile_id) or personas.get(DEFAULT_PERSONA_PROFILE_ID)
    if not profile:
        profile = PERSONA_PROFILES[DEFAULT_PERSONA_PROFILE_ID]
    return deepcopy(profile)


def update_persona_profile(state: Dict[str, Any], profile_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    profile_copy = deepcopy(profile)
    profile_copy["id"] = profile_id
    state.setdefault("personas", {})[profile_id] = profile_copy
    return save_platform_state(state)


def import_persona_profiles(state: Dict[str, Any], payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        for profile_id, profile in payload.items():
            if isinstance(profile, dict):
                update_persona_profile(state, profile_id, profile)
    return save_platform_state(state)


def update_prompt_settings(state: Dict[str, Any], prompt_settings: Dict[str, Any]) -> Dict[str, Any]:
    state["prompt_settings"] = {
        **state.get("prompt_settings", {}),
        **{key: str(value) for key, value in prompt_settings.items()},
    }
    return save_platform_state(state)


def update_api_settings(state: Dict[str, Any], api_settings: Dict[str, Any]) -> Dict[str, Any]:
    current = state.get("api_settings", {})
    next_settings = deepcopy(current)
    for key, value in api_settings.items():
        if key == "model_overrides" and isinstance(value, dict):
            next_settings[key] = value
        else:
            next_settings[key] = value
    state["api_settings"] = next_settings
    return save_platform_state(state)


def upsert_user(state: Dict[str, Any], user_data: Dict[str, Any]) -> Dict[str, Any]:
    users = state.setdefault("users", [])
    target_id = user_data.get("id") or f"user-{uuid4().hex[:8]}"
    record = {
        "id": target_id,
        "username": str(user_data.get("username", "")).strip(),
        "password": str(user_data.get("password", "")).strip(),
        "display_name": str(user_data.get("display_name", "")).strip() or str(user_data.get("username", "")).strip(),
        "role": str(user_data.get("role", "family")).strip() or "family",
        "persona_profile_id": str(user_data.get("persona_profile_id", DEFAULT_PERSONA_PROFILE_ID)).strip() or DEFAULT_PERSONA_PROFILE_ID,
        "default_view": str(user_data.get("default_view", "family")).strip() or "family",
        "enabled": bool(user_data.get("enabled", True)),
    }
    for index, user in enumerate(users):
        if user.get("id") == target_id:
            if not record["password"]:
                record["password"] = user.get("password", "")
            users[index] = record
            return save_platform_state(state)
    users.append(record)
    return save_platform_state(state)


def delete_user(state: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    state["users"] = [user for user in state.get("users", []) if user.get("id") != user_id]
    return save_platform_state(state)


def upsert_conversation_record(state: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    records = state.setdefault("conversation_records", [])
    session_id = record.get("session_id")
    for index, existing in enumerate(records):
        if existing.get("session_id") == session_id:
            records[index] = {**existing, **record, "updated_at": _now_iso()}
            return save_platform_state(state)
    records.append({**record, "updated_at": _now_iso()})
    return save_platform_state(state)


def delete_conversation_record(state: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    state["conversation_records"] = [item for item in state.get("conversation_records", []) if item.get("session_id") != session_id]
    return save_platform_state(state)
