from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from R_data import RewardCalculator
from dueling_dqn import DuelingDQNAgent
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


@dataclass
class RuntimeSession:
    algorithm: str
    script_file: str
    model_path: str
    turns: List[RuntimeTurn] = field(default_factory=list)
    filled_slots: Dict[str, List[str]] = field(default_factory=dict)
    current_script_id: int = 0
    current_step_index: int = 0
    transitions_used: int = 0
    last_deviation_high: bool = False
    started: bool = False

    def add_slot_values(self, slot_name: str, values: Iterable[str]) -> None:
        if slot_name not in self.filled_slots:
            self.filled_slots[slot_name] = []
        for value in values:
            if value not in self.filled_slots[slot_name]:
                self.filled_slots[slot_name].append(value)


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


class VirtualChildRLSystem:
    def __init__(
        self,
        algorithm: str = "dqn",
        script_file: Path | None = None,
        training_file: Path | None = None,
        model_dir: Path | None = None,
        policy_epochs: int = 180,
        policy_batch_size: int = 16,
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
        self.session = RuntimeSession(
            algorithm=self.policy.algorithm,
            script_file=str(self.script_file),
            model_path=str(self.policy.model_path),
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
            return self.current_step["child_dialogue"]

        next_script = self._pick_next_script(self.build_state_vector(False))
        self.session.current_script_id = int(next_script["script_id"])
        self.session.current_step_index = 0
        self.session.started = True
        logger.info("Started session with script_id=%s target_slot=%s", next_script["script_id"], next_script["target_slot"])
        return self.current_step["child_dialogue"]

    @property
    def current_script(self) -> Dict[str, Any]:
        return self.scripts_by_id[self.session.current_script_id]

    @property
    def current_step(self) -> Dict[str, Any]:
        return self.current_script["steps"][self.session.current_step_index]

    def _register_extracted_slots(self, extracted: Dict[str, List[str]]) -> None:
        for slot_name, values in extracted.items():
            self.session.add_slot_values(slot_name, values)

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

    def respond(self, elder_message: str) -> Dict[str, Any]:
        if not self.session.started:
            self.start_session()

        expected = self.current_step["expected_grandma_response"]
        assistant_message = self.current_step["child_dialogue"]
        similarity = self.similarity.score(expected, elder_message)
        deviation_level = self.similarity.deviation_level(similarity)
        extracted_slots = self.extractor.extract(elder_message)
        self._register_extracted_slots(extracted_slots)

        high_deviation = deviation_level >= 2
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
            deviation_level=deviation_level,
            extracted_slots=extracted_slots,
            transition_used=high_deviation,
        )
        self.session.turns.append(turn)

        current_state = self.build_state_vector(high_deviation)
        target_slot_complete = self.slot_completion_ratio(self.current_script["target_slot"]) >= 1.0

        if high_deviation:
            next_script = self._pick_next_script(current_state, prefer_new=True)
            self.session.current_script_id = int(next_script["script_id"])
            self.session.current_step_index = 0
            next_message = self._transition_message(next_script["target_slot"])
        else:
            next_message = None if target_slot_complete else self._advance_within_script()
            if next_message is None:
                next_message = self._switch_script(current_state, prefer_new=target_slot_complete)

        return {
            "assistant_message": next_message,
            "similarity": similarity,
            "deviation_level": deviation_level,
            "current_state": current_state,
            "filled_slots": self.session.filled_slots,
            "turn": turn.turn,
            "summary": self.build_summary_dict(),
        }

    def build_summary_dict(self) -> Dict[str, Any]:
        average_similarity = float(np.mean([turn.similarity for turn in self.session.turns])) if self.session.turns else 0.0
        average_deviation = float(np.mean([turn.deviation_level for turn in self.session.turns])) if self.session.turns else 0.0
        completion = {
            slot_name: {
                "filled_items": self.session.filled_slots.get(slot_name, []),
                "completion_ratio": round(self.slot_completion_ratio(slot_name), 3),
            }
            for slot_name in SLOT_ORDER
        }
        return {
            "algorithm": self.session.algorithm,
            "script_file": self.session.script_file,
            "model_path": self.session.model_path,
            "total_turns": len(self.session.turns),
            "average_similarity": round(average_similarity, 3),
            "average_deviation": round(average_deviation, 3),
            "transitions_used": self.session.transitions_used,
            "slot_completion": completion,
            "concerns": self._detect_concerns(),
            "next_focus_slots": self.get_incomplete_slots(),
        }

    def render_caregiver_summary(self) -> str:
        summary = self.build_summary_dict()
        lines = [
            "# 家屬照護摘要",
            "",
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
            }
            for turn in self.session.turns
        ]

    def build_ui_payload(self, latest_assistant_message: str | None = None) -> Dict[str, Any]:
        return {
            "algorithm": self.session.algorithm,
            "started": self.session.started,
            "latest_assistant_message": latest_assistant_message,
            "current_script_id": self.session.current_script_id,
            "current_target_slot": self.current_script["target_slot"] if self.session.started else None,
            "state_vector": self.build_state_vector(),
            "summary": self.build_summary_dict(),
            "summary_markdown": self.render_caregiver_summary(),
            "turns": self.serialize_turns(),
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
    )

    if args.mode == "interactive":
        paths = interactive_chat(system)
    else:
        paths = run_demo(system, load_demo_messages(args.demo_file))

    print(f"完成，成果檔案：{paths['transcript']} | {paths['summary']}")


if __name__ == "__main__":
    main()
