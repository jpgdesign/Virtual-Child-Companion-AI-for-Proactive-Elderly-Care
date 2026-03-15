"""
純對話模擬程式 - 只負責生成對話和收集資料
不計算獎勵，專注於對話品質和資料收集
"""

import os
import json
import random
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import openai
from openai import OpenAI
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DialogueState:
    """對話狀態"""
    current_step: int
    current_script_id: int
    filled_slots: Dict[str, List[str]]
    dialogue_history: List[Dict[str, str]]
    deviation_level: str
    script_context: List[Dict[str, str]]
    is_using_transition_script: bool

@dataclass
class DialogueResult:
    """純對話結果"""
    dialogue_history: List[Dict[str, str]]
    filled_slots: Dict[str, List[str]]
    rl_data: List[Dict]  # 強化學習資料
    total_turns: int
    average_similarity: float
    average_deviation: float
    transition_scripts_used: int

class DialogueSimulator:
    def __init__(self, 
                 script_file: str = None,
                 script_folder: str = None,
                 output_path: str = "outputs/dialogues/",
                 openai_key: str = None,
                 use_random_deviation: bool = False,
                 random_deviation_prob: float = 0.3):
        """初始化對話模擬器 - 純對話版本"""
        
        self.script_file = script_file
        self.script_folder = script_folder
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
        if openai_key:
            self.client = OpenAI(api_key=openai_key)
        else:
            raise ValueError("需要提供 OpenAI API 金鑰")
            
        self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # 槽位定義
        self.slot_definitions = {
            "用藥狀況": ["量血壓情況", "服藥情況", "身體不適", "用藥時間"],
            "睡眠狀態": ["起床時間", "睡眠時間", "小睡情況", "睡眠品質"],
            "作息活動": ["居家運動", "外出情況", "洗澡情況", "看電視情況"],
            "飲食狀況": ["三餐時間", "食物內容", "廚房使用", "冰箱使用"]
        }
        
        # 話題來源定義
        self.topic_sources = ["背景資料", "老人可能作息", "喜好興趣"]
        
        # 偏離度閾值設定
        self.deviation_thresholds = {
            0: 0.7,  # 偏離度0：相似度 ≥ 0.7
            1: 0.5,  # 偏離度1：相似度 0.5-0.7
            2: 0.3,  # 偏離度2：相似度 0.3-0.5
            3: 0.0   # 偏離度3：相似度 < 0.3
        }
        
        # 高偏離觸發閾值
        self.high_deviation_threshold = 2
        
        # 隨機偏離設定
        self.use_random_deviation = use_random_deviation
        self.random_deviation_prob = random_deviation_prob
        
        self.transition_scripts_used = 0
        self.original_scripts = []
        
        logger.info("純對話模擬器初始化完成")
        logger.info("偏離度分級：0(≥0.7), 1(0.5-0.7), 2(0.3-0.5), 3(<0.3)")
        
        if self.use_random_deviation:
            logger.info(f"啟用隨機偏離模式，概率: {self.random_deviation_prob:.1%}")

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """使用SBert計算文本相似度"""
        try:
            embedding1 = self.sbert_model.encode(text1, convert_to_tensor=True)
            embedding2 = self.sbert_model.encode(text2, convert_to_tensor=True)
            
            similarity = torch.nn.functional.cosine_similarity(embedding1.unsqueeze(0), 
                                                            embedding2.unsqueeze(0))
            
            return float(similarity[0])
        except Exception as e:
            logger.error(f"計算相似度時發生錯誤: {e}")
            return 0.0

    def similarity_to_deviation_level(self, similarity: float) -> int:
        """將相似度轉換為偏離度等級 (0-3)"""
        if similarity >= self.deviation_thresholds[0]:
            return 0
        elif similarity >= self.deviation_thresholds[1]:
            return 1
        elif similarity >= self.deviation_thresholds[2]:
            return 2
        else:
            return 3

    def is_high_deviation(self, deviation_level: int) -> bool:
        """判斷是否為高偏離（偏離度 ≥ 2）"""
        return deviation_level >= self.high_deviation_threshold

    def get_deviation_description(self, deviation_level: int) -> str:
        """獲取偏離度描述"""
        descriptions = {
            0: "相似",
            1: "輕微偏離", 
            2: "明顯偏離",
            3: "嚴重偏離"
        }
        return descriptions.get(deviation_level, "未知")

    def check_slot_filling(self, dialogue_text: str, target_slot: str) -> List[str]:
        """使用GPT檢查對話中填充了哪些槽位"""
        try:
            sub_slots = self.slot_definitions[target_slot]
            
            prompt = f"""請分析以下對話，判斷是否提到了以下資訊，只需回答有提到的項目：
對話內容：{dialogue_text}

需要檢查的資訊：
{', '.join(sub_slots)}

請只列出有提到的項目，每行一個，不要有其他文字。如果都沒提到，請回答"無"。"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=150
            )
            
            filled_slots = [slot.strip() for slot in 
                          response.choices[0].message.content.split('\n') 
                          if slot.strip() and slot.strip() != "無"]
            
            return filled_slots
            
        except Exception as e:
            logger.error(f"檢查槽位填充時發生錯誤: {e}")
            return []

    def generate_grandma_response_with_deviation(self, 
                                                dialogue_history: List[Dict[str, str]], 
                                                expected_response: str,
                                                target_deviation_level: int) -> str:
        """根據指定的偏離度生成奶奶的回應"""
        try:
            history_text = "\n".join([
                f"兒女：{turn['child']}\n奶奶：{turn['grandma']}" 
                for turn in dialogue_history[-3:] if turn['grandma']
            ])
            
            deviation_prompts = {
                0: "請生成一個與預期回應**高度相似**的回應，保持相同的意思和內容，只是措辭稍有不同。",
                1: "請生成一個與預期回應**輕微偏離**的回應，大致意思相同但增加一些個人色彩或細節變化。",
                2: "請生成一個與預期回應**明顯偏離**的回應，可能轉移到相關但不同的話題，或者表達不同的觀點。",
                3: "請生成一個與預期回應**嚴重偏離**的回應，可能完全離題或談論無關的事情，但仍然是合理的奶奶回應。"
            }
            
            deviation_prompt = deviation_prompts.get(target_deviation_level, deviation_prompts[0])
            
            prompt = f"""你現在是一位老奶奶，正在與兒女對話。

之前的對話：
{history_text}

預期的回應：
{expected_response}

偏離度要求：{target_deviation_level} (0=相似, 1=輕微偏離, 2=明顯偏離, 3=嚴重偏離)
{deviation_prompt}

請直接給出奶奶的回應，不要有任何額外說明。"""

            temperatures = {0: 0.3, 1: 0.7, 2: 1.0, 3: 1.2}
            temperature = temperatures.get(target_deviation_level, 0.7)

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=150
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"生成奶奶回應時發生錯誤: {e}")
            return expected_response

    def generate_target_deviation_level(self) -> int:
        """生成目標偏離度等級"""
        if self.use_random_deviation:
            return random.randint(0, 3)
        else:
            weights = [0.7, 0.2, 0.08, 0.02]
            return random.choices([0, 1, 2, 3], weights=weights)[0]

    def calculate_script_distance(self, current_context: List[str], script_turns: List[Dict]) -> float:
        """簡化版距離計算"""
        try:
            if not current_context or not script_turns:
                return 0.0
            
            context_length = len(current_context)
            script_length = len(script_turns)
            
            length_diff = abs(context_length - script_length)
            max_possible_diff = max(context_length, script_length, 1)
            distance = length_diff / max_possible_diff
            
            return distance
            
        except Exception as e:
            logger.error(f"計算劇本距離時發生錯誤: {e}")
            return 0.0

    def generate_transition_scripts(self, 
                                   dialogue_history: List[Dict[str, str]], 
                                   current_grandma_response: str) -> List[Dict]:
        """生成12個過渡劇本"""
        try:
            self.transition_scripts_used += 1
            
            recent_history = dialogue_history[-3:]
            history_summary = "\n".join([
                f"兒女: {turn['child']}\n奶奶: {turn['grandma']}"
                for turn in recent_history if turn['grandma']
            ])
            
            topic_sources = ["背景資料", "老人可能作息", "喜好興趣"]
            target_slots = ["用藥狀況", "睡眠狀態", "作息活動", "飲食狀況"]
            
            slot_topics = {
                "用藥狀況": ["量血壓情況", "服藥情況", "身體不適", "用藥時間"],
                "睡眠狀態": ["起床時間", "睡眠時間", "小睡情況", "睡眠品質"],
                "作息活動": ["居家運動", "外出情況", "洗澡情況", "看電視情況"],
                "飲食狀況": ["三餐時間", "食物內容", "廚房使用", "冰箱使用"]
            }
            
            transition_scripts = []
            script_id = 1
            
            for topic_source in topic_sources:
                for target_slot in target_slots:
                    related_topics = slot_topics.get(target_slot, ["日常生活"])
                    chosen_topic = random.choice(related_topics)
                    
                    script = self._generate_single_transition_script(
                        script_id=script_id,
                        topic_source=topic_source,
                        target_slot=target_slot,
                        chosen_topic=chosen_topic,
                        history_summary=history_summary,
                        current_grandma_response=current_grandma_response
                    )
                    
                    if script:
                        transition_scripts.append(script)
                    else:
                        fallback_script = self._create_fallback_transition_script(
                            script_id, topic_source, target_slot
                        )
                        transition_scripts.append(fallback_script)
                    
                    script_id += 1
            
            logger.info(f"生成了 {len(transition_scripts)} 個過渡劇本")
            return transition_scripts
            
        except Exception as e:
            logger.error(f"生成過渡劇本時發生錯誤: {e}")
            return self._create_all_fallback_scripts()

    def _generate_single_transition_script(self, 
                                         script_id: int,
                                         topic_source: str,
                                         target_slot: str,
                                         chosen_topic: str,
                                         history_summary: str,
                                         current_grandma_response: str) -> Dict:
        """生成單個過渡劇本"""
        try:
            prompt = f"""生成過渡劇本，引導對話從當前話題過渡到目標話題。

最近的對話歷史：
{history_summary}

奶奶剛才的回應：
{current_grandma_response}

目標槽位：{target_slot}
話題來源：{topic_source}
過渡話題：{chosen_topic}

請生成3個對話步驟：

第1步：
孫女：[溫和地回應奶奶的話]
預期奶奶：[配合的回應]

第2步：
孫女：[透過{chosen_topic}建立連結]
預期奶奶：[逐漸接近目標話題]

第3步：
孫女：[引導到{target_slot}相關話題]
預期奶奶：[為目標槽位對話做準備]
"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=400
            )
            
            content = response.choices[0].message.content
            steps = self._parse_transition_script_content(content)
            
            if len(steps) < 2:
                return None
            
            return {
                "script_id": script_id,
                "script_type": "過渡劇本",
                "source": topic_source,
                "target_slot": target_slot,
                "total_steps": len(steps),
                "steps": steps,
                "is_transition": True,
                "transition_topic": chosen_topic
            }
            
        except Exception as e:
            logger.error(f"生成過渡劇本 {script_id} 時發生錯誤: {e}")
            return None

    def _parse_transition_script_content(self, content: str) -> List[Dict]:
        """解析過渡劇本內容"""
        steps = []
        lines = content.split('\n')
        current_step = None
        child_dialogue = ""
        expected_response = ""

        for line in lines:
            line = line.strip()
            
            import re
            step_match = re.search(r'第(\d+)步', line)
            if step_match:
                if current_step is not None and child_dialogue and expected_response:
                    steps.append({
                        "step_number": current_step,
                        "child_dialogue": child_dialogue,
                        "expected_grandma_response": expected_response
                    })

                current_step = int(step_match.group(1))
                child_dialogue = ""
                expected_response = ""
                continue

            if line.startswith('孫女：') or line.startswith('孫女:'):
                child_dialogue = line.split('：', 1)[1].strip() if '：' in line else line.split(':', 1)[1].strip()
            elif line.startswith('預期奶奶：') or line.startswith('預期奶奶:'):
                expected_response = line.split('：', 1)[1].strip() if '：' in line else line.split(':', 1)[1].strip()

        if current_step is not None and child_dialogue and expected_response:
            steps.append({
                "step_number": current_step,
                "child_dialogue": child_dialogue,
                "expected_grandma_response": expected_response
            })

        return steps

    def _create_fallback_transition_script(self, script_id: int, topic_source: str, target_slot: str) -> Dict:
        """創建備用過渡劇本"""
        fallback_steps = [
            {
                "step_number": 1,
                "child_dialogue": "奶奶，我們剛才聊得很開心呢。",
                "expected_grandma_response": "是啊，和你聊天我總是很高興。"
            },
            {
                "step_number": 2,
                "child_dialogue": f"對了，我想關心一下您的{target_slot}，您最近怎麼樣？",
                "expected_grandma_response": "謝謝你的關心，我都還好。"
            }
        ]
        
        return {
            "script_id": script_id,
            "script_type": "備用過渡劇本",
            "source": topic_source,
            "target_slot": target_slot,
            "total_steps": len(fallback_steps),
            "steps": fallback_steps,
            "is_transition": True
        }

    def _create_all_fallback_scripts(self) -> List[Dict]:
        """創建所有12個備用過渡劇本"""
        topic_sources = ["背景資料", "老人可能作息", "喜好興趣"]
        target_slots = ["用藥狀況", "睡眠狀態", "作息活動", "飲食狀況"]
        
        scripts = []
        script_id = 1
        
        for topic_source in topic_sources:
            for target_slot in target_slots:
                script = self._create_fallback_transition_script(script_id, topic_source, target_slot)
                scripts.append(script)
                script_id += 1
        
        return scripts

    def is_all_slots_completely_filled(self, filled_slots: Dict[str, List[str]]) -> bool:
        """檢查是否所有槽位都已填滿"""
        for slot_name, sub_slots in self.slot_definitions.items():
            if not self.is_slot_completely_filled(slot_name, filled_slots):
                return False
        return True

    def is_slot_completely_filled(self, slot_name: str, filled_slots: Dict[str, List[str]]) -> bool:
        """檢查單個槽位是否完全填滿"""
        if slot_name not in filled_slots:
            return False
        
        filled_sub_slots = filled_slots[slot_name]
        required_sub_slots = self.slot_definitions[slot_name]
        
        invalid_responses = {
            "無", "無。", "無，", "無;", "無：", "無:",
            "", "沒有", "沒有提到", "沒有。", "沒有，",
            "未提及", "未提及。", "未提到", "未提到。",
            "none", "None", "NONE", "無相關資訊",
            "無此資訊", "無相關內容", "沒有相關資訊"
        }
        
        valid_filled = [slot.strip() for slot in filled_sub_slots 
                       if slot.strip() not in invalid_responses]
        
        filled_set = set(valid_filled)
        required_set = set(required_sub_slots)
        matched_slots = filled_set.intersection(required_set)
        
        return len(matched_slots) >= len(required_set)

    def get_incomplete_slots(self, filled_slots: Dict[str, List[str]]) -> List[str]:
        """獲取所有未完全填滿的槽位"""
        incomplete_slots = []
        for slot_name in self.slot_definitions.keys():
            if not self.is_slot_completely_filled(slot_name, filled_slots):
                incomplete_slots.append(slot_name)
        return incomplete_slots

    def select_script(self, available_scripts: List[Dict], filled_slots: Dict[str, List[str]]) -> Optional[Dict]:
        """選擇合適的劇本"""
        incomplete_slots = self.get_incomplete_slots(filled_slots)
        
        if not incomplete_slots:
            return None
        
        # 選擇完成度最低的槽位
        slot_progress = []
        for slot_name in incomplete_slots:
            filled_count = len(filled_slots.get(slot_name, []))
            required_count = len(self.slot_definitions[slot_name])
            completion_rate = filled_count / required_count
            slot_progress.append((slot_name, completion_rate))
        
        slot_progress.sort(key=lambda x: x[1])
        
        if random.random() < 0.8:
            target_slot = slot_progress[0][0]
        else:
            target_slot = random.choice(incomplete_slots)
        
        suitable_scripts = [
            script for script in available_scripts 
            if (script.get('target_slot') == target_slot and 
                script.get('script_id', 0) <= 12)
        ]
        
        if suitable_scripts:
            return random.choice(suitable_scripts)
        else:
            return None

    def select_transition_script(self, transition_scripts: List[Dict], filled_slots: Dict[str, List[str]]) -> Dict:
        """從12個過渡劇本中選擇一個"""
        incomplete_slots = self.get_incomplete_slots(filled_slots)
        
        if not incomplete_slots:
            return random.choice(transition_scripts)
        
        suitable_scripts = [
            script for script in transition_scripts
            if script.get('target_slot') in incomplete_slots
        ]
        
        if suitable_scripts:
            slot_progress = {}
            for slot_name in incomplete_slots:
                filled_count = len(filled_slots.get(slot_name, []))
                required_count = len(self.slot_definitions[slot_name])
                completion_rate = filled_count / required_count
                slot_progress[slot_name] = completion_rate
            
            min_completion_slot = min(slot_progress.keys(), key=lambda x: slot_progress[x])
            
            target_scripts = [
                script for script in suitable_scripts
                if script.get('target_slot') == min_completion_slot
            ]
            
            if target_scripts:
                return random.choice(target_scripts)
        
        return random.choice(transition_scripts)

    def load_scripts(self) -> List[Dict]:
        """載入劇本"""
        scripts = []
        if self.script_file:
            try:
                with open(self.script_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'scripts' in data:
                    scripts = data['scripts']
            except Exception as e:
                logger.error(f"載入劇本時發生錯誤: {e}")
        
        return scripts

    def simulate_dialogue(self, initial_script: Dict) -> DialogueResult:
        """模擬對話 - 只生成對話和收集資料"""
        # 初始化狀態
        state = DialogueState(
            current_step=0,
            current_script_id=initial_script['script_id'],
            filled_slots={},
            dialogue_history=[],
            deviation_level="low",
            script_context=[],
            is_using_transition_script=False
        )
        
        self.transition_scripts_used = 0
        
        rl_data = []  # 強化學習資料
        similarities = []
        deviations = []
        
        # 載入原始劇本
        available_scripts = self.load_scripts()
        if not available_scripts:
            logger.error("沒有可用的劇本")
            return DialogueResult(
                dialogue_history=[],
                filled_slots={},
                rl_data=[],
                total_turns=0,
                average_similarity=0.0,
                average_deviation=0.0,
                transition_scripts_used=0
            )
        
        self.original_scripts = available_scripts.copy()
        current_script = initial_script
        current_transition_scripts = []
        
        state.script_context = [{"utterance": step['expected_grandma_response']} 
                               for step in current_script['steps']]
        
        current_state = [0, 0, 0, 0, 0]
        turn_count = 0
        previous_script_distance = 0.0
        
        logger.info("開始純對話模擬")
        
        while True:
            turn_count += 1
            
            if turn_count % 10 == 0:
                logger.info(f"=== 第 {turn_count} 輪對話 ===")
            
            # 劇本選擇邏輯
            if state.current_step >= len(current_script['steps']):
                if state.is_using_transition_script:
                    new_script = self.select_script(self.original_scripts, state.filled_slots)
                    state.is_using_transition_script = False
                    current_transition_scripts = []
                else:
                    new_script = self.select_script(self.original_scripts, state.filled_slots)
                
                if not new_script:
                    new_script = random.choice(self.original_scripts)
                
                current_script = new_script
                state.current_step = 0
                state.script_context = [{"utterance": step['expected_grandma_response']} 
                                       for step in current_script['steps']]
            
            step = current_script['steps'][state.current_step]
            
            state.dialogue_history.append({
                'child': step['child_dialogue'],
                'grandma': ''
            })
            
            # 生成目標偏離度和奶奶回應
            target_deviation_level = self.generate_target_deviation_level()
            
            grandma_response = self.generate_grandma_response_with_deviation(
                state.dialogue_history,
                step['expected_grandma_response'],
                target_deviation_level
            )
            
            # 計算相似度和偏離度
            similarity = self.calculate_similarity(
                step['expected_grandma_response'],
                grandma_response
            )
            similarities.append(similarity)
            
            actual_deviation_level = self.similarity_to_deviation_level(similarity)
            deviations.append(actual_deviation_level)
            
            state.dialogue_history[-1]['grandma'] = grandma_response
            
            # 槽位填充檢查
            new_filled_slots = []
            if not state.is_using_transition_script and 'target_slot' in current_script:
                target_slot = current_script['target_slot']
                if target_slot in self.slot_definitions:
                    new_filled_slots = self.check_slot_filling(grandma_response, target_slot)
                    
                    if target_slot not in state.filled_slots:
                        state.filled_slots[target_slot] = []
                    
                    state.filled_slots[target_slot].extend(
                        slot for slot in new_filled_slots
                        if slot not in state.filled_slots[target_slot]
                    )
            
            # 更新狀態向量
            if (not state.is_using_transition_script and 
                'target_slot' in current_script and 
                current_script['target_slot'] in self.slot_definitions):
                try:
                    slot_index = list(self.slot_definitions.keys()).index(current_script['target_slot'])
                    current_state[slot_index] = 1 if new_filled_slots else 0
                except (ValueError, IndexError):
                    pass
            
            current_state[4] = 1 if self.is_high_deviation(actual_deviation_level) else 0
            
            # 計算劇本距離
            current_context = [turn['grandma'] for turn in state.dialogue_history if turn['grandma']]
            current_script_distance = self.calculate_script_distance(
                current_context, 
                state.script_context if state.script_context else [{"utterance": step['expected_grandma_response']}]
            )
            
            # 檢查是否完成所有槽位
            is_terminal = self.is_all_slots_completely_filled(state.filled_slots)
            
            # 創建action向量
            action_vector = [0.0] * 12
            if 1 <= current_script['script_id'] <= 12:
                action_vector[current_script['script_id'] - 1] = 1.0
            
            # 保存強化學習資料
            next_state = current_state.copy()
            
            rl_record = {
                'turn': turn_count,
                'state': current_state.copy(),
                'action': action_vector,
                'reward_requirements': {
                    'slot_filled': 1 if new_filled_slots else 0,  # Δst
                    'deviation_level': actual_deviation_level,  # 偏離度 (0-3)
                    'distance_improvement': max(0, previous_script_distance - current_script_distance),  # 距離改善
                    'is_terminal': is_terminal,  # 是否最後一輪
                    'total_turns': turn_count,  # 總輪數
                    'average_deviation': sum(deviations) / len(deviations) if deviations else 0  # 平均偏離度
                },
                'next_state': next_state,
                'script_info': {
                    'script_id': current_script['script_id'],
                    'script_type': current_script.get('script_type', 'original'),
                    'target_slot': current_script.get('target_slot', ''),
                    'source': current_script.get('source', ''),
                    'is_transition_script': state.is_using_transition_script
                },
                'dialogue_info': {
                    'expected_response': step['expected_grandma_response'],
                    'actual_response': grandma_response,
                    'similarity': similarity,
                    'deviation_description': self.get_deviation_description(actual_deviation_level)
                }
            }
            
            rl_data.append(rl_record)
            
            # 更新狀態
            state.current_step += 1
            current_state = next_state.copy()
            previous_script_distance = current_script_distance
            
            # 檢查對話是否結束
            if is_terminal:
                logger.info(f"🎉 所有槽位已完全填滿！對話在第 {turn_count} 輪結束")
                break
            
            # 高偏離處理：生成12個過渡劇本並選擇一個
            if self.is_high_deviation(actual_deviation_level) and not state.is_using_transition_script:
                logger.info(f"檢測到高偏離，生成12個過渡劇本")
                
                current_transition_scripts = self.generate_transition_scripts(
                    dialogue_history=state.dialogue_history,
                    current_grandma_response=grandma_response
                )
                
                if current_transition_scripts and len(current_transition_scripts) == 12:
                    selected_transition_script = self.select_transition_script(
                        current_transition_scripts, 
                        state.filled_slots
                    )
                    
                    current_script = selected_transition_script
                    state.current_step = 0
                    state.is_using_transition_script = True
                    state.script_context = [{"utterance": step['expected_grandma_response']} 
                                           for step in current_script['steps']]
                    
                    logger.info(f"切換到過渡劇本 {current_script['script_id']}")
        
        # 最終統計
        average_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        average_deviation = sum(deviations) / len(deviations) if deviations else 0.0
        
        logger.info(f"對話模擬完成！共 {turn_count} 輪")
        logger.info(f"平均相似度: {average_similarity:.3f}")
        logger.info(f"平均偏離度: {average_deviation:.3f}")
        logger.info(f"過渡劇本使用次數: {self.transition_scripts_used}")
        
        return DialogueResult(
            dialogue_history=state.dialogue_history,
            filled_slots=state.filled_slots,
            rl_data=rl_data,
            total_turns=turn_count,
            average_similarity=average_similarity,
            average_deviation=average_deviation,
            transition_scripts_used=self.transition_scripts_used
        )

    def save_dialogue_result(self, result: DialogueResult) -> Tuple[str, str]:
        """保存對話結果和RL資料"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 純對話檔案
        dialogue_file = os.path.join(self.output_path, f"pure_dialogue_{timestamp}.json")
        dialogue_data = {
            'timestamp': timestamp,
            'total_turns': result.total_turns,
            'dialogue_history': result.dialogue_history,
            'final_filled_slots': result.filled_slots,
            'statistics': {
                'average_similarity': result.average_similarity,
                'average_deviation': result.average_deviation,
                'transition_scripts_used': result.transition_scripts_used
            }
        }
        
        # RL資料檔案
        rl_file = os.path.join(self.output_path, f"rl_data_{timestamp}.json")
        rl_data = {
            'timestamp': timestamp,
            'total_turns': result.total_turns,
            'state_action_data': result.rl_data,
            'metadata': {
                'slot_definitions': self.slot_definitions,
                'deviation_thresholds': self.deviation_thresholds,
                'high_deviation_threshold': self.high_deviation_threshold
            }
        }
        
        try:
            with open(dialogue_file, 'w', encoding='utf-8') as f:
                json.dump(dialogue_data, f, ensure_ascii=False, indent=2)
            
            with open(rl_file, 'w', encoding='utf-8') as f:
                json.dump(rl_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"純對話已保存至: {dialogue_file}")
            logger.info(f"RL資料已保存至: {rl_file}")
            return dialogue_file, rl_file
            
        except Exception as e:
            logger.error(f"保存檔案時發生錯誤: {e}")
            return "", ""

def main():
    """主函數 - 純對話版本"""
    # 設置參數
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("Missing OPENAI_API_KEY environment variable.")
    script_file_path = r"C:\Users\sandy\Desktop\程式碼\outputs\scripts\grandma_session_20250713_221413\奶奶劇本\奶奶對話劇本_20250713_222053.json"  # 請替換為實際路徑
    
    # 初始化模擬器
    simulator = DialogueSimulator(
        script_file=script_file_path,
        output_path="outputs/dialogues/",
        openai_key=openai_key,
        use_random_deviation=True,
        random_deviation_prob=0.2
    )
    
    # 載入劇本
    scripts = simulator.load_scripts()
    if not scripts:
        logger.error("沒有找到可用的劇本")
        return
    
    logger.info(f"載入了 {len(scripts)} 個劇本")
    
    # 開始對話模擬
    initial_script = random.choice(scripts)
    logger.info(f"開始對話模擬，初始劇本 ID: {initial_script['script_id']}")
    
    result = simulator.simulate_dialogue(initial_script)
    
    # 保存結果
    dialogue_file, rl_file = simulator.save_dialogue_result(result)
    
    # 輸出摘要
    print(f"\n=== 純對話模擬完成！ ===")
    print(f"總對話輪數: {result.total_turns}")
    print(f"過渡劇本使用次數: {result.transition_scripts_used}")
    print(f"平均相似度: {result.average_similarity:.3f}")
    print(f"平均偏離度: {result.average_deviation:.3f}")
    print(f"已填充槽位: {result.filled_slots}")
    print(f"純對話已保存至: {dialogue_file}")
    print(f"RL資料已保存至: {rl_file}")

if __name__ == "__main__":
    main()
