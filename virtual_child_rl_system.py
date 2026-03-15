from __future__ import annotations

import argparse
import json
import logging
import random
import re
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from R_data import RewardCalculator
from dueling_dqn import DuelingDQNAgent
from llm_runtime import (
    DEFAULT_ANALYSIS_PRESET,
    DEFAULT_GENERATION_PRESET,
    HybridLLMOrchestrator,
    LLMAnalysisResult,
    build_endpoint_config,
    available_model_presets,
)
from persona_profiles import (
    DEFAULT_PERSONA_PROFILE_ID,
    available_persona_profiles,
    get_persona_profile,
)
from tabular_q_learning import TabularQLearningAgent

try:
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional runtime acceleration
    SentenceTransformer = None
    torch = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


ALGORITHM_ALIASES = {
    "dqn": "dqn",
    "dueling_dqn": "dqn",
    "q": "q_learning",
    "q_learning": "q_learning",
    "qlearning": "q_learning",
    "tabular_q": "q_learning",
}

SLOT_ORDER = ["用藥狀況", "睡眠狀態", "作息活動", "飲食狀況"]

SLOT_DEFINITIONS = {
    "用藥狀況": ["量血壓情況", "服藥情況", "身體不適", "用藥時間"],
    "睡眠狀態": ["起床時間", "睡眠時間", "小睡情況", "睡眠品質"],
    "作息活動": ["上廁所情況", "居家運動", "外出情況", "洗澡情況", "喝水情況", "看電視情況"],
    "飲食狀況": ["三餐時間", "食物內容", "廚房使用", "冰箱使用"],
}

FAST_PENDING_QUESTIONS = {
    "用藥狀況": {
        "量血壓情況": "對了，您今天有量血壓嗎？大概是多少呢？",
        "服藥情況": "今天的藥都有按時吃嗎？",
        "身體不適": "除了剛剛提到的，現在身體還有哪裡不舒服嗎？",
        "用藥時間": "那您今天大概是幾點吃藥的呢？",
    },
    "睡眠狀態": {
        "起床時間": "您今天大概幾點起床呢？",
        "睡眠時間": "昨晚大概睡了多久呢？",
        "小睡情況": "白天有沒有再小睡一下？",
        "睡眠品質": "昨晚整體睡得還安穩嗎？",
    },
    "作息活動": {
        "上廁所情況": "今天上廁所都還順嗎？",
        "居家運動": "今天有沒有在家活動一下，像伸展或簡單運動？",
        "外出情況": "今天有沒有出門走走呢？",
        "洗澡情況": "今天有沒有洗澡或簡單梳洗？",
        "喝水情況": "今天水有喝得夠嗎？",
        "看電視情況": "今天有沒有看喜歡的節目？",
    },
    "飲食狀況": {
        "三餐時間": "今天這餐大概是幾點吃的呢？",
        "食物內容": "除了剛剛說的，這餐還有搭配什麼嗎？",
        "廚房使用": "今天有自己進廚房準備東西嗎？",
        "冰箱使用": "今天有自己去冰箱拿東西嗎？",
    },
}

FAST_TOPIC_ACKS = [
    (re.compile(r"暈|頭暈|不舒服|難受|痛|沒力", re.I), "聽起來您剛剛有點不太舒服，我有在留意。"),
    (re.compile(r"血壓|量血壓", re.I), "有記得量血壓這點很好，我也比較放心。"),
    (re.compile(r"吃|早餐|午餐|晚餐|稀飯|麵|飯|菜|喝", re.I), "有把東西吃一點、喝一點都很重要。"),
    (re.compile(r"睡|起床|午睡|失眠", re.I), "有休息到真的很重要，我想先陪您慢慢聊。"),
    (re.compile(r"公園|散步|走走|氣功|運動|出門", re.I), "有活動一下或出門走走，聽起來還不錯。"),
]

SLOT_PATTERNS = {
    "用藥狀況": {
        "量血壓情況": [r"血壓", r"血糖", r"量.*血壓", r"測.*血糖"],
        "服藥情況": [r"吃藥", r"服藥", r"藥都有", r"控制血糖的藥", r"忘記吃藥", r"藥放"],
        "身體不適": [r"不舒服", r"頭暈", r"胸悶", r"無力", r"疼", r"痠", r"喘"],
        "用藥時間": [r"早餐後", r"午餐後", r"晚餐後", r"飯後", r"早上.*吃藥", r"晚上.*吃藥", r"\d+點.*吃藥"],
    },
    "睡眠狀態": {
        "起床時間": [r"\d+點.*起床", r"自然醒", r"早上.*起來", r"五點.*起床", r"六點.*起床"],
        "睡眠時間": [r"\d+點.*睡", r"上床", r"昨晚.*睡", r"晚上.*睡", r"十點.*睡"],
        "小睡情況": [r"午睡", r"小睡", r"瞇一下", r"睡個半小時"],
        "睡眠品質": [r"睡得.*好", r"睡不好", r"失眠", r"淺眠", r"多夢", r"很精神", r"睡得還不錯"],
    },
    "作息活動": {
        "上廁所情況": [r"上廁所", r"廁所"],
        "居家運動": [r"運動", r"氣功", r"瑜珈", r"拉筋", r"體操"],
        "外出情況": [r"散步", r"公園", r"市場", r"出門", r"超市", r"河濱"],
        "洗澡情況": [r"洗澡", r"沐浴"],
        "喝水情況": [r"喝水", r"開水", r"溫水", r"綠茶", r"豆漿", r"茶"],
        "看電視情況": [r"看電視", r"八點檔", r"包青天", r"節目", r"新聞"],
    },
    "飲食狀況": {
        "三餐時間": [r"早餐", r"午餐", r"晚餐", r"早上吃", r"中午吃", r"晚上吃"],
        "食物內容": [r"粥", r"燕麥", r"水果", r"地瓜", r"醬菜", r"豆花", r"青菜", r"魚", r"肉", r"湯", r"麵", r"飯"],
        "廚房使用": [r"煮飯", r"煮菜", r"做菜", r"下廚", r"廚房", r"開火"],
        "冰箱使用": [r"冰箱", r"冷藏", r"拿飲料", r"拿水果"],
    },
}

TRANSITION_TEMPLATES = {
    "用藥狀況": [
        "我先記下您剛剛說的，聽起來最近身體狀況值得多留意。我也想順著關心一下，您今天量血壓或吃藥時還順不順利？",
        "謝謝您跟我分享，我有把這些放在心上。像您平常照顧身體很有規律，今天用藥或量測血壓的情況如何呢？",
    ],
    "睡眠狀態": [
        "我懂，那種感覺真的會影響整天精神。我昨晚也有點晚睡，所以特別想問問您昨晚休息得怎麼樣，大概幾點睡呢？",
        "您說得很真實，我有聽進去。最近天氣忽冷忽熱，我也會擔心睡眠，您這幾天睡得安穩嗎？",
    ],
    "作息活動": [
        "原來如此，我好像已經能想像您的日常節奏了。今天天氣不錯，您有出去走走、做運動，或在家活動一下嗎？",
        "謝謝您慢慢跟我說，我很想多知道您的日常。您今天有沒有散步、練氣功，或看一下喜歡的節目呢？",
    ],
    "飲食狀況": [
        "聽您這樣說，我突然想到今天吃飯這件事也很重要。您今天三餐吃得怎麼樣，有沒有吃到自己喜歡的東西？",
        "我記下來了，謝謝您願意跟我聊。我剛剛也在想晚點要煮什麼，您今天早餐或午餐吃了些什麼呢？",
    ],
}

CONCERN_PATTERNS = {
    "疑似漏藥或服藥不穩定": [r"忘記吃藥", r"沒吃藥", r"不想吃藥", r"常常忘"],
    "睡眠品質偏差": [r"睡不好", r"失眠", r"淺眠", r"一直醒", r"多夢"],
    "食慾或進食偏少": [r"吃不下", r"沒胃口", r"不太想吃", r"只喝", r"沒吃什麼"],
    "身體不適需要留意": [r"頭暈", r"胸悶", r"喘", r"很不舒服", r"疼", r"痛"],
    "飲水量可能不足": [r"不想喝水", r"很少喝水", r"忘記喝水"],
}

DEFAULT_DEMO_MESSAGES = [
    "哎呀，我今天五點多就起床了，天氣涼涼的很適合去公園練氣功。",
    "運動完我會餓，所以早餐吃了地瓜粥和一點醬菜，還喝了溫水。",
    "我吃完早餐大概一兩個小時就會吃控制血糖的藥，藥都放在餐桌上比較不會忘記。",
    "中午有時候會小睡半小時，晚上差不多十點就上床，最近睡得還不錯。",
    "下午我會去散步，回來洗個澡，再看看包青天，這樣心情很好。",
    "偶爾也會量一下血壓，覺得身體沒有什麼特別不舒服。",
]


def normalize_algorithm(name: str) -> str:
    normalized = ALGORITHM_ALIASES.get(name.lower())
    if not normalized:
        raise ValueError(f"Unsupported algorithm: {name}")
    return normalized


def find_latest_file(pattern: str) -> Path:
    matches = sorted(Path(".").glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"Could not find a file matching {pattern}")
    return matches[0]


@dataclass
class RuntimeTurn:
    turn: int
    assistant_message: str
    elder_message: str
    expected_response: str
    script_id: int
    target_slot: str
    similarity: float
    deviation_level: int
    extracted_slots: Dict[str, List[str]]
    transition_used: bool
    emotion_label: str = ""
    emotion_intensity: float = 0.0
    llm_analysis_summary: str = ""
    llm_concerns: List[str] = field(default_factory=list)
    generated_by_llm: bool = False


@dataclass
class RuntimeSession:
    algorithm: str
    script_file: str
    model_path: str
    persona_profile_id: str = DEFAULT_PERSONA_PROFILE_ID
    persona_profile: Dict[str, Any] = field(default_factory=dict)
    prompt_settings: Dict[str, Any] = field(default_factory=dict)
    turns: List[RuntimeTurn] = field(default_factory=list)
    filled_slots: Dict[str, List[str]] = field(default_factory=dict)
    slot_value_details: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    current_script_id: int = 0
    current_step_index: int = 0
    transitions_used: int = 0
    last_deviation_high: bool = False
    started: bool = False
    latest_emotion: Dict[str, Any] = field(default_factory=dict)
    latest_analysis: Dict[str, Any] = field(default_factory=dict)
    llm_enabled: bool = False
    llm_status: Dict[str, Any] = field(default_factory=dict)
    latest_assistant_message: str = ""
    analysis_token: int = 0
    background_processing: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def add_slot_values(self, slot_name: str, values: Iterable[str]) -> None:
        if slot_name not in self.filled_slots:
            self.filled_slots[slot_name] = []
        for value in values:
            if value not in self.filled_slots[slot_name]:
                self.filled_slots[slot_name].append(value)

    def add_slot_value_detail(self, slot_name: str, item_name: str, value: str) -> None:
        if not value:
            return
        self.slot_value_details.setdefault(slot_name, {}).setdefault(item_name, [])
        if value not in self.slot_value_details[slot_name][item_name]:
            self.slot_value_details[slot_name][item_name].append(value)


class SimilarityScorer:
    def __init__(self) -> None:
        self.model = None
        self.using_embeddings = SentenceTransformer is not None and torch is not None
        if self.using_embeddings:
            try:
                self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                logger.info("Loaded multilingual sentence-transformer for deviation detection.")
            except Exception as exc:  # pragma: no cover - depends on local model cache
                self.using_embeddings = False
                logger.warning("Falling back to token overlap similarity: %s", exc)

    def score(self, expected: str, actual: str) -> float:
        if self.using_embeddings and self.model is not None and torch is not None:
            embedding_a = self.model.encode(expected, convert_to_tensor=True, show_progress_bar=False)
            embedding_b = self.model.encode(actual, convert_to_tensor=True, show_progress_bar=False)
            similarity = torch.nn.functional.cosine_similarity(
                embedding_a.unsqueeze(0),
                embedding_b.unsqueeze(0),
            )
            return float(similarity[0])

        expected_tokens = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", expected))
        actual_tokens = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", actual))
        if not expected_tokens or not actual_tokens:
            return 0.0
        return len(expected_tokens & actual_tokens) / len(expected_tokens | actual_tokens)

    @staticmethod
    def deviation_level(similarity: float) -> int:
        if similarity >= 0.7:
            return 0
        if similarity >= 0.5:
            return 1
        if similarity >= 0.3:
            return 2
        return 3


class KeywordSlotExtractor:
    def __init__(self, slot_patterns: Dict[str, Dict[str, List[str]]]) -> None:
        self.slot_patterns = slot_patterns

    def extract(self, text: str) -> Dict[str, List[str]]:
        matches: Dict[str, List[str]] = {}
        for slot_name, definitions in self.slot_patterns.items():
            current: List[str] = []
            for sub_slot, patterns in definitions.items():
                if any(re.search(pattern, text) for pattern in patterns):
                    current.append(sub_slot)
            if current:
                matches[slot_name] = current
        return matches


class PolicyRuntime:
    def __init__(
        self,
        algorithm: str,
        script_file: Path,
        training_file: Path,
        model_dir: Path,
        epochs: int = 180,
        batch_size: int = 16,
    ) -> None:
        self.algorithm = normalize_algorithm(algorithm)
        self.script_file = script_file
        self.training_file = training_file
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / (
            "runtime_dqn_model.pt" if self.algorithm == "dqn" else "runtime_q_learning_model.json"
        )
        self.metadata_path = self.model_dir / f"runtime_{self.algorithm}_metadata.json"
        self.epochs = epochs
        self.batch_size = batch_size
        self.agent = self._build_agent()
        self.records: List[Dict[str, Any]] = []
        self._load_or_train()

    def _build_agent(self) -> DuelingDQNAgent | TabularQLearningAgent:
        if self.algorithm == "dqn":
            return DuelingDQNAgent(
                state_dim=5,
                action_dim=12,
                learning_rate=0.001,
                gamma=0.99,
                epsilon=1.0,
                epsilon_min=0.1,
                epsilon_decay=0.995,
                target_update=20,
                memory_capacity=10_000,
                batch_size=self.batch_size,
            )

        return TabularQLearningAgent(
            state_dim=5,
            action_dim=12,
            alpha=0.12,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.995,
            memory_capacity=10_000,
            batch_size=self.batch_size,
        )

    def _build_policy_records(self) -> List[Dict[str, Any]]:
        payload = json.loads(self.training_file.read_text(encoding="utf-8"))
        raw_turns = payload.get("state_action_data")
        if not raw_turns:
            raise ValueError(f"{self.training_file} does not contain state_action_data")

        reward_records = RewardCalculator().calculate_all_rewards(raw_turns)
        progress = {slot_name: 0 for slot_name in SLOT_ORDER}
        previous_high_deviation = 0
        records: List[Dict[str, Any]] = []

        for raw_turn, reward_record in zip(raw_turns, reward_records):
            state = [progress[slot_name] for slot_name in SLOT_ORDER] + [previous_high_deviation]
            target_slot = raw_turn.get("script_info", {}).get("target_slot", "")
            slot_filled = int(reward_record.get("reward_requirements", {}).get("slot_filled", 0))
            if slot_filled and target_slot in progress:
                progress[target_slot] = 1

            current_high_deviation = (
                1 if reward_record.get("reward_requirements", {}).get("deviation_level", 0) >= 2 else 0
            )
            next_state = [progress[slot_name] for slot_name in SLOT_ORDER] + [current_high_deviation]

            records.append(
                {
                    "state": state,
                    "action": raw_turn["action"],
                    "reward": float(reward_record["reward"]),
                    "next_state": next_state,
                    "done": bool(reward_record.get("reward_requirements", {}).get("is_terminal", False)),
                }
            )
            previous_high_deviation = current_high_deviation

        return records

    def _load_or_train(self) -> None:
        if self.model_path.exists():
            self.agent.load_model(str(self.model_path))
            logger.info("Loaded %s runtime policy from %s", self.algorithm, self.model_path)
            return

        self.records = self._build_policy_records()

        for record in self.records:
            self.agent.memory.push(
                record["state"],
                record["action"],
                record["reward"],
                record["next_state"],
                record["done"],
            )

        for _ in range(self.epochs):
            self.agent.train(self.batch_size)

        self.agent.save_model(str(self.model_path))
        metadata = {
            "algorithm": self.algorithm,
            "training_file": str(self.training_file),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "records": len(self.records),
            "average_reward": float(np.mean([record["reward"] for record in self.records])),
        }
        self.metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Trained %s runtime policy and saved model to %s", self.algorithm, self.model_path)

    def choose_script_id(self, state: Sequence[int], allowed_script_ids: Sequence[int]) -> int:
        candidate_ids = list(dict.fromkeys(int(script_id) for script_id in allowed_script_ids))
        if not candidate_ids:
            raise ValueError("No candidate scripts available for policy selection.")

        q_values = self.agent.get_q_values(state)
        ranked = sorted(candidate_ids, key=lambda script_id: float(q_values[script_id - 1]), reverse=True)
        return ranked[0]

    def ranked_script_ids(self, state: Sequence[int], allowed_script_ids: Sequence[int]) -> List[int]:
        candidate_ids = list(dict.fromkeys(int(script_id) for script_id in allowed_script_ids))
        q_values = self.agent.get_q_values(state)
        return sorted(candidate_ids, key=lambda script_id: float(q_values[script_id - 1]), reverse=True)

    def ranked_candidates(self, state: Sequence[int], allowed_script_ids: Sequence[int], limit: int = 3) -> List[Dict[str, Any]]:
        candidate_ids = self.ranked_script_ids(state, allowed_script_ids)
        q_values = self.agent.get_q_values(state)
        return [
            {
                "script_id": script_id,
                "q_value": round(float(q_values[script_id - 1]), 4),
            }
            for script_id in candidate_ids[:limit]
        ]


class VirtualChildRLSystem:
    def __init__(
        self,
        algorithm: str = "dqn",
        script_file: Path | None = None,
        training_file: Path | None = None,
        model_dir: Path | None = None,
        policy_epochs: int = 180,
        policy_batch_size: int = 16,
        llm_enabled: bool = True,
        analysis_preset: str = DEFAULT_ANALYSIS_PRESET,
        generation_preset: str = DEFAULT_GENERATION_PRESET,
        persona_profile_id: str = DEFAULT_PERSONA_PROFILE_ID,
        persona_profile_data: Dict[str, Any] | None = None,
        prompt_settings: Dict[str, Any] | None = None,
    ) -> None:
        self.script_file = script_file or find_latest_file("grandma_session_*/*/奶奶對話劇本_*.json")
        self.training_file = training_file or find_latest_file("rl_data_*.json")
        self.policy = PolicyRuntime(
            algorithm=algorithm,
            script_file=self.script_file,
            training_file=self.training_file,
            model_dir=model_dir or Path("artifacts/runtime_models"),
            epochs=policy_epochs,
            batch_size=policy_batch_size,
        )
        payload = json.loads(self.script_file.read_text(encoding="utf-8"))
        self.scripts: List[Dict[str, Any]] = payload["scripts"]
        self.scripts_by_id = {int(script["script_id"]): script for script in self.scripts}
        self.script_ids_by_slot: Dict[str, List[int]] = {}
        for script in self.scripts:
            self.script_ids_by_slot.setdefault(script["target_slot"], []).append(int(script["script_id"]))

        self.similarity = SimilarityScorer()
        self.extractor = KeywordSlotExtractor(SLOT_PATTERNS)
        self.llm_enabled = llm_enabled
        self.llm = None
        llm_status: Dict[str, Any] = {"enabled": False}
        if llm_enabled:
            self.llm = HybridLLMOrchestrator(
                analysis_config=build_endpoint_config(analysis_preset, fallback=DEFAULT_ANALYSIS_PRESET),
                generation_config=build_endpoint_config(generation_preset, fallback=DEFAULT_GENERATION_PRESET),
            )
            llm_status = self.llm.status_dict()
        persona_profile = deepcopy(persona_profile_data) if persona_profile_data else get_persona_profile(persona_profile_id)
        self.session = RuntimeSession(
            algorithm=self.policy.algorithm,
            script_file=str(self.script_file),
            model_path=str(self.policy.model_path),
            persona_profile_id=persona_profile["id"],
            persona_profile=persona_profile,
            prompt_settings=dict(prompt_settings or {}),
            llm_enabled=llm_enabled,
            llm_status=llm_status,
        )

    def build_state_vector(self, high_deviation: bool | None = None) -> List[int]:
        flags = []
        for slot_name in SLOT_ORDER:
            slot_values = self.session.filled_slots.get(slot_name, [])
            flags.append(1 if slot_values else 0)
        last_flag = self.session.last_deviation_high if high_deviation is None else high_deviation
        flags.append(1 if last_flag else 0)
        return flags

    def slot_completion_ratio(self, slot_name: str) -> float:
        required = SLOT_DEFINITIONS[slot_name]
        filled = self.session.filled_slots.get(slot_name, [])
        return len(set(filled) & set(required)) / len(required)

    def get_incomplete_slots(self) -> List[str]:
        return [slot_name for slot_name in SLOT_ORDER if self.slot_completion_ratio(slot_name) < 1.0]

    def _allowed_script_ids(self) -> List[int]:
        incomplete_slots = self.get_incomplete_slots()
        if not incomplete_slots:
            return [int(script["script_id"]) for script in self.scripts]

        script_ids: List[int] = []
        for slot_name in incomplete_slots:
            script_ids.extend(self.script_ids_by_slot.get(slot_name, []))
        return script_ids

    def _pick_next_script(self, state: Sequence[int], prefer_new: bool = False) -> Dict[str, Any]:
        allowed_script_ids = self._allowed_script_ids()
        ranked_ids = self.policy.ranked_script_ids(state, allowed_script_ids)
        for script_id in ranked_ids:
            if not prefer_new or script_id != self.session.current_script_id:
                return self.scripts_by_id[script_id]
        return self.scripts_by_id[ranked_ids[0]]

    def start_session(self) -> str:
        if self.session.started:
            return self.session.latest_assistant_message or self._apply_persona_voice(self.current_step["child_dialogue"], add_address=True)

        next_script = self._pick_next_script(self.build_state_vector(False))
        self.session.current_script_id = int(next_script["script_id"])
        self.session.current_step_index = 0
        self.session.started = True
        logger.info("Started session with script_id=%s target_slot=%s", next_script["script_id"], next_script["target_slot"])
        opening_message = self._apply_persona_voice(self.current_step["child_dialogue"], add_address=True)
        self.session.latest_assistant_message = opening_message
        return opening_message

    @property
    def current_script(self) -> Dict[str, Any]:
        return self.scripts_by_id[self.session.current_script_id]

    @property
    def current_step(self) -> Dict[str, Any]:
        return self.current_script["steps"][self.session.current_step_index]

    @property
    def persona_profile(self) -> Dict[str, Any]:
        return self.session.persona_profile

    def _preferred_elder_address(self) -> str:
        child_profile = self.persona_profile.get("child", {})
        return str(child_profile.get("preferred_elder_address", "")).strip()

    def _apply_persona_voice(self, message: str, *, add_address: bool = False) -> str:
        text = str(message or "").strip()
        if not text:
            return text
        elder_address = self._preferred_elder_address()
        if not elder_address:
            return text
        text = re.sub(
            r"^(奶奶|阿嬤|阿媽|外婆|婆婆|媽媽|媽|爸爸|爸|伯父|伯母)[，、,:： ]*",
            f"{elder_address}，",
            text,
            count=1,
        )
        if add_address and not text.startswith(elder_address):
            return f"{elder_address}，{text}"
        return text

    def _persona_overview(self) -> Dict[str, Any]:
        child = self.persona_profile.get("child", {})
        elder = self.persona_profile.get("elder", {})
        relationship = self.persona_profile.get("relationship", {})
        return {
            "id": self.session.persona_profile_id,
            "label": self.persona_profile.get("label", self.session.persona_profile_id),
            "child_name": child.get("name", ""),
            "child_role": child.get("role", ""),
            "child_role_detail": child.get("role_detail", ""),
            "child_address": child.get("preferred_elder_address", ""),
            "elder_name": elder.get("name", ""),
            "elder_role": elder.get("role", ""),
            "family_mapping": relationship.get("family_mapping", ""),
            "dynamic": relationship.get("dynamic", ""),
            "guidance_style": relationship.get("guidance_style", ""),
        }

    def _recent_turn_context(self, limit: int = 4) -> List[Dict[str, Any]]:
        return [
            {
                "turn": turn.turn,
                "assistant_message": turn.assistant_message,
                "elder_message": turn.elder_message,
                "target_slot": turn.target_slot,
                "deviation_level": turn.deviation_level,
                "emotion_label": turn.emotion_label,
            }
            for turn in self.session.turns[-limit:]
        ]

    def _ranked_candidate_context(self, state: Sequence[int], limit: int = 3) -> List[Dict[str, Any]]:
        allowed_script_ids = self._allowed_script_ids()
        ranked = self.policy.ranked_candidates(state, allowed_script_ids, limit=limit)
        enriched: List[Dict[str, Any]] = []
        for candidate in ranked:
            script = self.scripts_by_id.get(int(candidate["script_id"]))
            if not script:
                continue
            enriched.append(
                {
                    **candidate,
                    "target_slot": script["target_slot"],
                    "reference_reply": script["steps"][0]["child_dialogue"],
                }
            )
        return enriched

    def _pick_pending_item(self, slot_name: str) -> str:
        for item in SLOT_DEFINITIONS[slot_name]:
            if item not in self.session.filled_slots.get(slot_name, []):
                return item
        return SLOT_DEFINITIONS[slot_name][0]

    def _build_fast_acknowledgement(self, elder_message: str) -> str:
        for pattern, acknowledgement in FAST_TOPIC_ACKS:
            if pattern.search(elder_message):
                return acknowledgement
        return "我有在聽，您慢慢說就好。"

    def _match_fast_acknowledgement(self, elder_message: str) -> tuple[re.Pattern[str] | None, str]:
        for pattern, acknowledgement in FAST_TOPIC_ACKS:
            if pattern.search(elder_message):
                return pattern, acknowledgement
        return None, "我有在聽，您慢慢說就好。"

    def _build_fast_guidance(self, target_slot: str) -> str:
        pending_item = self._pick_pending_item(target_slot)
        question = FAST_PENDING_QUESTIONS.get(target_slot, {}).get(pending_item)
        if question:
            return question
        return f"我也想多了解一下您今天的{target_slot}，可以再和我說一點嗎？"

    def _extract_reference_question(self, reference_reply: str) -> str:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[。！？!?])", reference_reply or "")
            if sentence.strip()
        ]
        for sentence in reversed(sentences):
            if "？" in sentence or "嗎" in sentence:
                return sentence
        return ""

    def _blend_generated_reply(self, elder_message: str, fast_reply_hint: str, generated_reply: str) -> str:
        candidate = (generated_reply or "").strip()
        if not candidate:
            return fast_reply_hint
        topic_pattern, acknowledgement = self._match_fast_acknowledgement(elder_message)
        if topic_pattern is None or topic_pattern.search(candidate):
            return candidate
        follow_up = self._extract_reference_question(candidate) or candidate
        if follow_up == acknowledgement:
            return fast_reply_hint
        return f"{acknowledgement}{follow_up}"

    def _build_fast_reply(
        self,
        elder_message: str,
        target_slot: str,
        reference_reply: str,
        extracted_slots: Dict[str, List[str]],
    ) -> str:
        acknowledgement = self._build_fast_acknowledgement(elder_message)
        same_topic = bool(extracted_slots.get(target_slot))
        guidance = self._build_fast_guidance(target_slot)
        next_prompt = guidance
        if same_topic:
            reference_question = self._extract_reference_question(reference_reply)
            if reference_question and len(reference_question) <= 28:
                next_prompt = reference_question
        if same_topic:
            return f"{acknowledgement}順著您剛剛提到的，{next_prompt}"
        return f"{acknowledgement}我先記下來，也順便幫您確認一下，{next_prompt}"

    def _register_extracted_slots(self, extracted: Dict[str, List[str]]) -> None:
        for slot_name, values in extracted.items():
            self.session.add_slot_values(slot_name, values)

    def _register_llm_slot_candidates(self, analysis: LLMAnalysisResult) -> None:
        for candidate in analysis.slot_candidates:
            if candidate.slot not in SLOT_DEFINITIONS:
                continue
            if candidate.item not in SLOT_DEFINITIONS[candidate.slot]:
                continue
            self.session.add_slot_values(candidate.slot, [candidate.item])
            self.session.add_slot_value_detail(candidate.slot, candidate.item, candidate.value)

    def _merge_concerns(self, llm_concerns: Sequence[str]) -> List[str]:
        combined = self._detect_concerns()
        for concern in llm_concerns:
            if concern not in combined:
                combined.append(concern)
        return combined

    def _transition_message(self, target_slot: str) -> str:
        templates = TRANSITION_TEMPLATES.get(target_slot, TRANSITION_TEMPLATES["作息活動"])
        return random.choice(templates)

    def _advance_within_script(self) -> Optional[str]:
        next_index = self.session.current_step_index + 1
        if next_index < len(self.current_script["steps"]):
            self.session.current_step_index = next_index
            return self.current_step["child_dialogue"]
        return None

    def _switch_script(self, state: Sequence[int], prefer_new: bool = False) -> str:
        next_script = self._pick_next_script(state, prefer_new=prefer_new)
        self.session.current_script_id = int(next_script["script_id"])
        self.session.current_step_index = 0
        return self.current_step["child_dialogue"]

    def _detect_concerns(self) -> List[str]:
        combined_text = "\n".join(turn.elder_message for turn in self.session.turns)
        concerns = []
        for label, patterns in CONCERN_PATTERNS.items():
            if any(re.search(pattern, combined_text) for pattern in patterns):
                concerns.append(label)
        return concerns

    def _background_process_turn(
        self,
        *,
        token: int,
        turn_index: int,
        elder_message: str,
        source_script: Dict[str, Any],
        source_step: Dict[str, Any],
        selected_script: Dict[str, Any],
        reference_reply: str,
        fast_reply_hint: str,
        current_target_slot: str,
        target_slot_items: List[str],
        pending_items: List[str],
        recent_turns: List[Dict[str, Any]],
        filled_slots_snapshot: Dict[str, List[str]],
        slot_value_details_snapshot: Dict[str, Dict[str, List[str]]],
        regex_extracted: Dict[str, List[str]],
        similarity_score: float,
        rule_deviation: int,
        ranked_candidates: List[Dict[str, Any]],
        transition_mode: bool,
    ) -> None:
        if self.llm is None:
            return

        try:
            analysis = self.llm.analyze_turn(
                persona_context=self.persona_profile,
                prompt_overrides=self.session.prompt_settings,
                elder_message=elder_message,
                current_script=source_script,
                current_step=source_step,
                recent_turns=recent_turns,
                filled_slots=filled_slots_snapshot,
                slot_definitions=SLOT_DEFINITIONS,
                regex_extracted=regex_extracted,
                similarity_score=similarity_score,
                similarity_deviation=rule_deviation,
                ranked_candidates=ranked_candidates,
            )
            generation = self.llm.generate_reply(
                persona_context=self.persona_profile,
                prompt_overrides=self.session.prompt_settings,
                elder_message=elder_message,
                selected_script=selected_script,
                reference_reply=reference_reply,
                fast_reply_hint=fast_reply_hint,
                current_target_slot=current_target_slot,
                target_slot_items=target_slot_items,
                pending_items=pending_items,
                filled_slots=filled_slots_snapshot,
                slot_value_details=slot_value_details_snapshot,
                analysis=analysis,
                recent_turns=recent_turns,
                ranked_candidates=ranked_candidates,
                transition_mode=transition_mode,
            )

            with self.session.lock:
                if token != self.session.analysis_token:
                    return

                self._register_llm_slot_candidates(analysis)
                self.session.latest_emotion = {
                    "label": analysis.emotion.label,
                    "intensity": analysis.emotion.intensity,
                    "evidence": analysis.emotion.evidence,
                }
                self.session.latest_analysis = analysis.to_dict()
                self.session.latest_analysis["status"] = "completed"
                refined_reply = self._blend_generated_reply(elder_message, fast_reply_hint, generation.reply)
                refined_reply = self._apply_persona_voice(refined_reply, add_address=True)
                generation_dict = generation.to_dict()
                generation_dict["reply"] = refined_reply
                self.session.latest_analysis["generation"] = generation_dict
                self.session.latest_assistant_message = refined_reply or self.session.latest_assistant_message
                self.session.background_processing = False

                if 0 <= turn_index < len(self.session.turns):
                    turn = self.session.turns[turn_index]
                    turn.deviation_level = max(turn.deviation_level, analysis.deviation_level or 0)
                    turn.emotion_label = analysis.emotion.label
                    turn.emotion_intensity = analysis.emotion.intensity
                    turn.llm_analysis_summary = analysis.summary
                    turn.llm_concerns = list(analysis.concerns)

        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Background LLM processing failed: %s", exc)
            with self.session.lock:
                if token != self.session.analysis_token:
                    return
                self.session.background_processing = False
                self.session.latest_analysis = {
                    "status": "failed",
                    "summary": "",
                    "error": str(exc),
                    "emotion": {"label": "分析失敗", "intensity": 0.0, "evidence": ""},
                    "slot_candidates": [],
                    "concerns": [],
                    "deviation_level": rule_deviation,
                    "model_label": self.session.llm_status.get("analysis", {}).get("model", ""),
                }

    def respond_fast(self, elder_message: str) -> Dict[str, Any]:
        if not self.session.started:
            self.start_session()

        with self.session.lock:
            source_script = dict(self.current_script)
            source_step = {
                "child_dialogue": self.current_step["child_dialogue"],
                "expected_grandma_response": self.current_step["expected_grandma_response"],
            }
            expected = source_step["expected_grandma_response"]
            assistant_message = self._apply_persona_voice(source_step["child_dialogue"], add_address=True)
            similarity = self.similarity.score(expected, elder_message)
            rule_deviation = self.similarity.deviation_level(similarity)
            extracted_slots = self.extractor.extract(elder_message)
            self._register_extracted_slots(extracted_slots)

            high_deviation = rule_deviation >= 2
            self.session.last_deviation_high = high_deviation
            if high_deviation:
                self.session.transitions_used += 1

            turn = RuntimeTurn(
                turn=len(self.session.turns) + 1,
                assistant_message=assistant_message,
                elder_message=elder_message,
                expected_response=expected,
                script_id=self.session.current_script_id,
                target_slot=self.current_script["target_slot"],
                similarity=similarity,
                deviation_level=rule_deviation,
                extracted_slots=extracted_slots,
                transition_used=high_deviation,
            )
            self.session.turns.append(turn)
            turn_index = len(self.session.turns) - 1

            current_state = self.build_state_vector(high_deviation)
            target_slot_complete = self.slot_completion_ratio(self.current_script["target_slot"]) >= 1.0
            transition_mode = False

            if high_deviation:
                next_script = self._pick_next_script(current_state, prefer_new=True)
                self.session.current_script_id = int(next_script["script_id"])
                self.session.current_step_index = 0
                reference_reply = self._transition_message(next_script["target_slot"])
                transition_mode = True
            else:
                reference_reply = None if target_slot_complete else self._advance_within_script()
                if reference_reply is None:
                    reference_reply = self._switch_script(current_state, prefer_new=target_slot_complete)
            if reference_reply:
                reference_reply = self._apply_persona_voice(reference_reply, add_address=True)

            current_target_slot = self.current_script["target_slot"]
            fast_reply = self._build_fast_reply(
                elder_message,
                current_target_slot,
                reference_reply or "",
                extracted_slots,
            )
            fast_reply = self._apply_persona_voice(fast_reply, add_address=True)
            self.session.latest_assistant_message = fast_reply

            ranked_candidates = self._ranked_candidate_context(current_state)
            pending_items = [
                item
                for item in SLOT_DEFINITIONS[current_target_slot]
                if item not in self.session.filled_slots.get(current_target_slot, [])
            ]
            filled_slots_snapshot = {key: list(values) for key, values in self.session.filled_slots.items()}
            slot_value_details_snapshot = {
                slot_name: {item: list(values) for item, values in item_map.items()}
                for slot_name, item_map in self.session.slot_value_details.items()
            }
            recent_turns = self._recent_turn_context()

            if self.llm is not None:
                self.session.analysis_token += 1
                token = self.session.analysis_token
                self.session.background_processing = True
                self.session.latest_emotion = {"label": "分析中", "intensity": 0.0, "evidence": ""}
                self.session.latest_analysis = {
                    "status": "pending",
                    "summary": "背景分析中，稍後會補上情緒、填槽與較自然的回覆。",
                    "emotion": self.session.latest_emotion,
                    "slot_candidates": [],
                    "concerns": [],
                    "deviation_level": rule_deviation,
                    "recommended_focus": current_target_slot,
                    "model_label": self.session.llm_status.get("analysis", {}).get("model", ""),
                }
            else:
                token = 0
                self.session.background_processing = False
                self.session.latest_emotion = {}
                self.session.latest_analysis = {
                    "status": "disabled",
                    "summary": "",
                    "emotion": {},
                    "slot_candidates": [],
                    "concerns": [],
                    "deviation_level": rule_deviation,
                }

        if self.llm is not None:
            worker = threading.Thread(
                target=self._background_process_turn,
                kwargs={
                    "token": token,
                    "turn_index": turn_index,
                    "elder_message": elder_message,
                    "source_script": source_script,
                    "source_step": source_step,
                    "selected_script": dict(self.current_script),
                    "reference_reply": reference_reply or fast_reply,
                    "fast_reply_hint": fast_reply,
                    "current_target_slot": current_target_slot,
                    "target_slot_items": list(SLOT_DEFINITIONS[current_target_slot]),
                    "pending_items": list(pending_items),
                    "recent_turns": list(recent_turns),
                    "filled_slots_snapshot": filled_slots_snapshot,
                    "slot_value_details_snapshot": slot_value_details_snapshot,
                    "regex_extracted": {key: list(values) for key, values in extracted_slots.items()},
                    "similarity_score": similarity,
                    "rule_deviation": rule_deviation,
                    "ranked_candidates": list(ranked_candidates),
                    "transition_mode": transition_mode,
                },
                daemon=True,
            )
            worker.start()

        return {
            "assistant_message": fast_reply,
            "similarity": similarity,
            "deviation_level": rule_deviation,
            "current_state": current_state,
            "filled_slots": self.session.filled_slots,
            "turn": turn.turn,
            "summary": self.build_summary_dict(),
            "analysis": self.session.latest_analysis,
            "background_processing": self.session.background_processing,
        }

    def respond(self, elder_message: str) -> Dict[str, Any]:
        if not self.session.started:
            self.start_session()

        expected = self.current_step["expected_grandma_response"]
        assistant_message = self._apply_persona_voice(self.current_step["child_dialogue"], add_address=True)
        similarity = self.similarity.score(expected, elder_message)
        rule_deviation = self.similarity.deviation_level(similarity)
        extracted_slots = self.extractor.extract(elder_message)
        self._register_extracted_slots(extracted_slots)

        high_deviation = rule_deviation >= 2
        self.session.last_deviation_high = high_deviation
        if high_deviation:
            self.session.transitions_used += 1

        turn = RuntimeTurn(
            turn=len(self.session.turns) + 1,
            assistant_message=assistant_message,
            elder_message=elder_message,
            expected_response=expected,
            script_id=self.session.current_script_id,
            target_slot=self.current_script["target_slot"],
            similarity=similarity,
            deviation_level=rule_deviation,
            extracted_slots=extracted_slots,
            transition_used=high_deviation,
        )
        self.session.turns.append(turn)

        current_state = self.build_state_vector(high_deviation)
        target_slot_complete = self.slot_completion_ratio(self.current_script["target_slot"]) >= 1.0
        transition_mode = False

        if high_deviation:
            next_script = self._pick_next_script(current_state, prefer_new=True)
            self.session.current_script_id = int(next_script["script_id"])
            self.session.current_step_index = 0
            next_message = self._transition_message(next_script["target_slot"])
            transition_mode = True
        else:
            next_message = None if target_slot_complete else self._advance_within_script()
            if next_message is None:
                next_message = self._switch_script(current_state, prefer_new=target_slot_complete)
        if next_message:
            next_message = self._apply_persona_voice(next_message, add_address=True)

        analysis = LLMAnalysisResult()
        self.session.latest_emotion = {
            "label": analysis.emotion.label,
            "intensity": analysis.emotion.intensity,
            "evidence": analysis.emotion.evidence,
        }
        self.session.latest_analysis = analysis.to_dict()

        ranked_candidates_after = self._ranked_candidate_context(current_state)
        pending_items = [
            item
            for item in SLOT_DEFINITIONS[self.current_script["target_slot"]]
            if item not in self.session.filled_slots.get(self.current_script["target_slot"], [])
        ]

        if self.llm is not None and next_message:
            try:
                if self.llm.use_fused_turns:
                    analysis, generation = self.llm.analyze_and_generate_turn(
                        persona_context=self.persona_profile,
                        prompt_overrides=self.session.prompt_settings,
                        elder_message=elder_message,
                        current_script=self.scripts_by_id[turn.script_id],
                        current_step={
                            "child_dialogue": assistant_message,
                            "expected_grandma_response": expected,
                        },
                        selected_script=self.current_script,
                        reference_reply=next_message,
                        current_target_slot=self.current_script["target_slot"],
                        target_slot_items=SLOT_DEFINITIONS[self.current_script["target_slot"]],
                        pending_items=pending_items,
                        recent_turns=self._recent_turn_context(),
                        filled_slots=self.session.filled_slots,
                        regex_extracted=extracted_slots,
                        similarity_score=similarity,
                        rule_deviation=rule_deviation,
                        ranked_candidates=ranked_candidates_after,
                        transition_mode=transition_mode,
                    )
                else:
                    analysis = self.llm.analyze_turn(
                        persona_context=self.persona_profile,
                        prompt_overrides=self.session.prompt_settings,
                        elder_message=elder_message,
                        current_script=self.scripts_by_id[turn.script_id],
                        current_step={
                            "child_dialogue": assistant_message,
                            "expected_grandma_response": expected,
                        },
                        recent_turns=self._recent_turn_context(),
                        filled_slots=self.session.filled_slots,
                        slot_definitions=SLOT_DEFINITIONS,
                        regex_extracted=extracted_slots,
                        similarity_score=similarity,
                        similarity_deviation=rule_deviation,
                        ranked_candidates=ranked_candidates_after,
                    )
                    generation = self.llm.generate_reply(
                        persona_context=self.persona_profile,
                        prompt_overrides=self.session.prompt_settings,
                        elder_message=elder_message,
                        selected_script=self.current_script,
                        reference_reply=next_message,
                        fast_reply_hint=next_message,
                        current_target_slot=self.current_script["target_slot"],
                        target_slot_items=SLOT_DEFINITIONS[self.current_script["target_slot"]],
                        pending_items=pending_items,
                        filled_slots=self.session.filled_slots,
                        slot_value_details=self.session.slot_value_details,
                        analysis=analysis,
                        recent_turns=self._recent_turn_context(),
                        ranked_candidates=ranked_candidates_after,
                        transition_mode=transition_mode,
                    )

                self._register_llm_slot_candidates(analysis)
                self.session.latest_emotion = {
                    "label": analysis.emotion.label,
                    "intensity": analysis.emotion.intensity,
                    "evidence": analysis.emotion.evidence,
                }
                self.session.latest_analysis = analysis.to_dict()
                self.session.latest_analysis["generation"] = generation.to_dict()
                turn.deviation_level = max(rule_deviation, analysis.deviation_level or 0)
                turn.emotion_label = analysis.emotion.label
                turn.emotion_intensity = analysis.emotion.intensity
                turn.llm_analysis_summary = analysis.summary
                turn.llm_concerns = list(analysis.concerns)
                if generation.reply:
                    next_message = self._apply_persona_voice(generation.reply, add_address=True)
                    turn.generated_by_llm = True
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("LLM processing failed, using reference reply: %s", exc)

        return {
            "assistant_message": next_message,
            "similarity": similarity,
            "deviation_level": turn.deviation_level,
            "current_state": current_state,
            "filled_slots": self.session.filled_slots,
            "turn": turn.turn,
            "summary": self.build_summary_dict(),
            "analysis": self.session.latest_analysis,
        }

    def build_summary_dict(self) -> Dict[str, Any]:
        average_similarity = float(np.mean([turn.similarity for turn in self.session.turns])) if self.session.turns else 0.0
        average_deviation = float(np.mean([turn.deviation_level for turn in self.session.turns])) if self.session.turns else 0.0
        completion = {
            slot_name: {
                "filled_items": self.session.filled_slots.get(slot_name, []),
                "completion_ratio": round(self.slot_completion_ratio(slot_name), 3),
                "value_notes": self.session.slot_value_details.get(slot_name, {}),
            }
            for slot_name in SLOT_ORDER
        }
        return {
            "algorithm": self.session.algorithm,
            "script_file": self.session.script_file,
            "model_path": self.session.model_path,
            "persona_profile": self.persona_profile,
            "persona_overview": self._persona_overview(),
            "total_turns": len(self.session.turns),
            "average_similarity": round(average_similarity, 3),
            "average_deviation": round(average_deviation, 3),
            "transitions_used": self.session.transitions_used,
            "slot_completion": completion,
            "concerns": self._merge_concerns(self.session.latest_analysis.get("concerns", [])),
            "next_focus_slots": self.get_incomplete_slots(),
            "latest_emotion": self.session.latest_emotion,
            "latest_analysis_summary": self.session.latest_analysis.get("summary", ""),
            "llm_status": self.session.llm_status,
        }

    def render_caregiver_summary(self) -> str:
        summary = self.build_summary_dict()
        persona_overview = summary.get("persona_overview") or {}
        lines = [
            "# 家屬照護摘要",
            "",
            f"- 家庭畫像：`{persona_overview.get('label', '')}`",
            f"- 家庭關係：{persona_overview.get('family_mapping', '')}",
            f"- 演算法：`{summary['algorithm']}`",
            f"- 總對話輪數：`{summary['total_turns']}`",
            f"- 平均相似度：`{summary['average_similarity']}`",
            f"- 平均偏離度：`{summary['average_deviation']}`",
            f"- 轉場次數：`{summary['transitions_used']}`",
            "",
            "## 槽位蒐集進度",
        ]
        for slot_name in SLOT_ORDER:
            slot_info = summary["slot_completion"][slot_name]
            filled_items = "、".join(slot_info["filled_items"]) if slot_info["filled_items"] else "尚未蒐集"
            lines.append(
                f"- {slot_name}：完成度 `{slot_info['completion_ratio']:.0%}`，已蒐集 `{filled_items}`"
            )

        concerns = summary["concerns"] or ["目前對話未偵測到明顯高風險訊號"]
        lines.extend(["", "## 需要留意", *[f"- {item}" for item in concerns]])

        next_focus = summary["next_focus_slots"] or ["目前四大槽位皆已有資料，可改追蹤趨勢變化"]
        lines.extend(["", "## 下一步建議", *[f"- {item}" for item in next_focus]])
        latest_emotion = summary.get("latest_emotion") or {}
        if latest_emotion:
            lines.extend(
                [
                    "",
                    "## LLM 分析",
                    f"- 最新情緒：`{latest_emotion.get('label', '平穩')}`",
                    f"- 情緒強度：`{latest_emotion.get('intensity', 0.0)}`",
                ]
            )

        latest_analysis_summary = summary.get("latest_analysis_summary")
        if latest_analysis_summary:
            lines.append(f"- 分析摘要：{latest_analysis_summary}")

        llm_status = summary.get("llm_status") or {}
        if llm_status.get("enabled"):
            lines.append(f"- 分析模型：`{llm_status.get('analysis', {}).get('model', '')}`")
            lines.append(f"- 生成模型：`{llm_status.get('generation', {}).get('model', '')}`")

        return "\n".join(lines)

    def serialize_turns(self) -> List[Dict[str, Any]]:
        return [
            {
                "turn": turn.turn,
                "assistant_message": turn.assistant_message,
                "elder_message": turn.elder_message,
                "expected_response": turn.expected_response,
                "script_id": turn.script_id,
                "target_slot": turn.target_slot,
                "similarity": round(turn.similarity, 3),
                "deviation_level": turn.deviation_level,
                "transition_used": turn.transition_used,
                "extracted_slots": turn.extracted_slots,
                "emotion_label": turn.emotion_label,
                "emotion_intensity": round(turn.emotion_intensity, 3),
                "llm_analysis_summary": turn.llm_analysis_summary,
                "llm_concerns": turn.llm_concerns,
                "generated_by_llm": turn.generated_by_llm,
            }
            for turn in self.session.turns
        ]

    def build_ui_payload(self, latest_assistant_message: str | None = None) -> Dict[str, Any]:
        return {
            "algorithm": self.session.algorithm,
            "started": self.session.started,
            "latest_assistant_message": latest_assistant_message or self.session.latest_assistant_message,
            "persona_profile_id": self.session.persona_profile_id,
            "persona_profile": self.persona_profile,
            "current_script_id": self.session.current_script_id,
            "current_target_slot": self.current_script["target_slot"] if self.session.started else None,
            "state_vector": self.build_state_vector(),
            "summary": self.build_summary_dict(),
            "summary_markdown": self.render_caregiver_summary(),
            "turns": self.serialize_turns(),
            "latest_analysis": self.session.latest_analysis,
            "llm_status": self.session.llm_status,
            "available_model_presets": available_model_presets(),
            "available_persona_profiles": available_persona_profiles(),
            "background_processing": self.session.background_processing,
        }

    def save_session(self, output_dir: Path | None = None) -> Dict[str, str]:
        output_dir = output_dir or Path("artifacts/runtime_demo")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcript_path = output_dir / f"runtime_session_{timestamp}.json"
        summary_path = output_dir / f"caregiver_summary_{timestamp}.md"

        transcript = {
            "summary": self.build_summary_dict(),
            "turns": self.serialize_turns(),
        }
        transcript_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_path.write_text(self.render_caregiver_summary(), encoding="utf-8")
        return {"transcript": str(transcript_path), "summary": str(summary_path)}


def run_demo(system: VirtualChildRLSystem, messages: Sequence[str]) -> Dict[str, str]:
    assistant = system.start_session()
    print(f"虛擬兒女：{assistant}")
    for message in messages:
        print(f"長者：{message}")
        result = system.respond(message)
        print(f"虛擬兒女：{result['assistant_message']}")
    paths = system.save_session()
    print(f"Transcript saved to: {paths['transcript']}")
    print(f"Caregiver summary saved to: {paths['summary']}")
    return paths


def interactive_chat(system: VirtualChildRLSystem) -> Dict[str, str]:
    print("輸入 `summary` 可查看摘要，輸入 `exit` 離開。")
    print(f"虛擬兒女：{system.start_session()}")

    while True:
        elder_message = input("長者：").strip()
        if not elder_message:
            continue
        if elder_message.lower() in {"exit", "quit"}:
            break
        if elder_message.lower() == "summary":
            print(system.render_caregiver_summary())
            continue

        result = system.respond(elder_message)
        print(f"虛擬兒女：{result['assistant_message']}")

    return system.save_session()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL-driven virtual child dialogue runtime.")
    parser.add_argument("--algorithm", default="dqn", choices=sorted(ALGORITHM_ALIASES))
    parser.add_argument("--mode", default="demo", choices=["demo", "interactive"])
    parser.add_argument("--script-file", type=Path, default=None)
    parser.add_argument("--training-file", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/runtime_models"))
    parser.add_argument("--policy-epochs", type=int, default=180)
    parser.add_argument("--policy-batch-size", type=int, default=16)
    parser.add_argument("--demo-file", type=Path, default=None, help="Optional JSON file with demo_messages array.")
    parser.add_argument("--disable-llm", action="store_true", help="Disable hybrid LLM analysis/generation.")
    parser.add_argument("--analysis-preset", default=DEFAULT_ANALYSIS_PRESET, choices=sorted(available_model_presets()))
    parser.add_argument("--generation-preset", default=DEFAULT_GENERATION_PRESET, choices=sorted(available_model_presets()))
    parser.add_argument("--persona-profile", default=DEFAULT_PERSONA_PROFILE_ID, choices=sorted(available_persona_profiles()))
    return parser.parse_args()


def load_demo_messages(demo_file: Path | None) -> List[str]:
    if not demo_file:
        return list(DEFAULT_DEMO_MESSAGES)
    payload = json.loads(demo_file.read_text(encoding="utf-8"))
    messages = payload.get("demo_messages")
    if not isinstance(messages, list) or not all(isinstance(item, str) for item in messages):
        raise ValueError(f"{demo_file} must contain a demo_messages string array")
    return messages


def main() -> None:
    args = parse_args()
    system = VirtualChildRLSystem(
        algorithm=args.algorithm,
        script_file=args.script_file,
        training_file=args.training_file,
        model_dir=args.model_dir,
        policy_epochs=args.policy_epochs,
        policy_batch_size=args.policy_batch_size,
        llm_enabled=not args.disable_llm,
        analysis_preset=args.analysis_preset,
        generation_preset=args.generation_preset,
        persona_profile_id=args.persona_profile,
    )

    if args.mode == "interactive":
        paths = interactive_chat(system)
    else:
        paths = run_demo(system, load_demo_messages(args.demo_file))

    print(f"完成，成果檔案：{paths['transcript']} | {paths['summary']}")


if __name__ == "__main__":
    main()
