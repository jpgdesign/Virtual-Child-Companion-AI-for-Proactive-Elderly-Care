"""
獎勵計算程式 - 讀取對話資料並計算獎勵，生成最終的(s,a,r,s')資料
"""
import os
import json
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RewardParameters:
    """獎勵函數參數"""
    # 即時獎勵參數
    alpha: float = 15.0   # 填槽獎勵係數
    phi: float = 5.0      # 距離度調整係數
    theta: float = 3.0    # 距離獎勵係數
    
    # 終端獎勵參數
    R_goal: float = 300.0  # 目標完成獎勵
    mu: float = 30.0       # 對話長度懲罰強度
    T_safe: int = 8        # 安全輪數
    delta: float = 10.0    # 平均偏離度懲罰強度

class RewardCalculator:
    def __init__(self, params: RewardParameters = None):
        """初始化獎勵計算器"""
        self.params = params if params else RewardParameters()
        
        # R_case 獎勵表
        self.R_case_table = {
            # 沒填槽位的情況 - 加重懲罰
            (0, 0): -2.0,   # 沒填槽 + 無偏離
            (0, 1): -3.0,   # 沒填槽 + 小偏離
            (0, 2): -5.0,   # 沒填槽 + 中偏離
            (0, 3): -8.0,   # 沒填槽 + 大偏離
            
            # 成功填槽的情況 - 大幅提高獎勵
            (1, 0): 5.0,    # 填槽 + 無偏離（完美情況）
            (1, 1): 3.0,    # 填槽 + 小偏離（仍然很好）
            (1, 2): 1.0,    # 填槽 + 中偏離（至少是正獎勵）
            (1, 3): 0.0     # 填槽 + 大偏離（不懲罰成功）
        }
        
        logger.info("獎勵計算器初始化完成")
        logger.info(f"獎勵參數: α={self.params.alpha}, φ={self.params.phi}, θ={self.params.theta}")
        logger.info(f"終端參數: R_goal={self.params.R_goal}, μ={self.params.mu}, T_safe={self.params.T_safe}, δ={self.params.delta}")

    def get_R_case(self, slot_filled: int, deviation_level: int) -> float:
        """根據填槽情況和偏離度獲取R_case獎勵"""
        key = (slot_filled, deviation_level)
        reward = self.R_case_table.get(key, 0.0)
        
        logger.debug(f"R_case查表: 填槽={slot_filled}, 偏離度={deviation_level} -> 獎勵={reward}")
        return reward

    def calculate_immediate_reward(self, 
                                 slot_filled: int,           # 是否填槽 (0或1)
                                 deviation_level: int,       # 偏離度等級 (0-3)
                                 distance_improvement: float # 距離改善量
                                 ) -> Tuple[float, Dict[str, float]]:
        """
        計算即時獎勵
        rt = [α∆St] + [φRcase(∆St, deviation)] + [θ(1-∆St)max(0, distance_improvement)]
        """
        try:
            # 第一項：槽位填充獎勵 [α∆St]
            slot_reward = self.params.alpha * slot_filled
            
            # 第二項：偏離度調整獎勵 [φRcase(∆St, deviation)]
            r_case = self.get_R_case(slot_filled, deviation_level)
            deviation_reward = self.params.phi * r_case
            
            # 第三項：劇本距離獎勵 [θ(1-∆St)max(0, distance_improvement)]
            # 只有在沒填槽時(1-∆St=1)才計算距離改善獎勵
            distance_reward = self.params.theta * (1 - slot_filled) * max(0.0, distance_improvement)
            
            # 總即時獎勵
            total_reward = slot_reward + deviation_reward + distance_reward
            
            # 獎勵分解
            breakdown = {
                'slot_reward': slot_reward,
                'deviation_reward': deviation_reward,
                'distance_reward': distance_reward,
                'total_immediate': total_reward
            }
            
            logger.debug(f"即時獎勵計算:")
            logger.debug(f"  槽位獎勵: α({self.params.alpha}) × ∆S({slot_filled}) = {slot_reward}")
            logger.debug(f"  偏離獎勵: φ({self.params.phi}) × Rcase({r_case}) = {deviation_reward}")
            logger.debug(f"  距離獎勵: θ({self.params.theta}) × (1-∆S({1-slot_filled})) × {distance_improvement:.3f} = {distance_reward}")
            logger.debug(f"  總即時獎勵: {total_reward:.3f}")
            
            return total_reward, breakdown
            
        except Exception as e:
            logger.error(f"計算即時獎勵時發生錯誤: {e}")
            return 0.0, {}

    def calculate_terminal_reward(self, 
                                total_turns: int, 
                                average_deviation: float,
                                all_slots_filled: bool = True) -> Tuple[float, Dict[str, float]]:
        """
        計算終端獎勵
        rterminal = [Rgoal] - [μ log(1 + T - Tsafe)] - [δ deviation]
        """
        try:
            # 第一項：目標完成獎勵 [Rgoal]
            goal_reward = self.params.R_goal if all_slots_filled else 0.0
            
            # 第二項：對話長度懲罰 [μ log(1 + T - Tsafe)]
            if total_turns > self.params.T_safe:
                length_penalty = self.params.mu * np.log(1 + total_turns - self.params.T_safe)
            else:
                length_penalty = 0.0
            
            # 第三項：平均偏離度懲罰 [δ deviation]
            deviation_penalty = self.params.delta * average_deviation
            
            # 總終端獎勵
            terminal_reward = goal_reward - length_penalty - deviation_penalty
            
            # 獎勵分解
            breakdown = {
                'goal_reward': goal_reward,
                'length_penalty': -length_penalty,
                'deviation_penalty': -deviation_penalty,
                'total_terminal': terminal_reward
            }
            
            logger.info(f"終端獎勵計算:")
            logger.info(f"  目標完成: {goal_reward} (所有槽位填滿: {all_slots_filled})")
            logger.info(f"  長度懲罰: -{length_penalty:.3f} (輪數: {total_turns}, 安全輪數: {self.params.T_safe})")
            logger.info(f"  偏離懲罰: -{deviation_penalty:.3f} (平均偏離度: {average_deviation:.3f})")
            logger.info(f"  終端獎勵: {terminal_reward:.3f}")
            
            return terminal_reward, breakdown
            
        except Exception as e:
            logger.error(f"計算終端獎勵時發生錯誤: {e}")
            return 0.0, {}

    def calculate_all_rewards(self, rl_data: List[Dict]) -> List[Dict]:
        """計算所有獎勵並生成最終的(s,a,r,s')資料"""
        logger.info(f"開始計算 {len(rl_data)} 輪對話的獎勵")
        
        final_data = []
        total_reward = 0.0
        
        for i, turn_data in enumerate(rl_data):
            try:
                # 提取獎勵計算所需資訊
                reward_req = turn_data['reward_requirements']
                slot_filled = reward_req['slot_filled']
                deviation_level = reward_req['deviation_level']
                distance_improvement = reward_req['distance_improvement']
                is_terminal = reward_req['is_terminal']
                total_turns = reward_req['total_turns']
                average_deviation = reward_req['average_deviation']
                
                # 計算即時獎勵
                immediate_reward, immediate_breakdown = self.calculate_immediate_reward(
                    slot_filled=slot_filled,
                    deviation_level=deviation_level,
                    distance_improvement=distance_improvement
                )
                
                # 如果是終端狀態，加上終端獎勵
                terminal_reward = 0.0
                terminal_breakdown = {}
                if is_terminal:
                    terminal_reward, terminal_breakdown = self.calculate_terminal_reward(
                        total_turns=total_turns,
                        average_deviation=average_deviation,
                        all_slots_filled=True
                    )
                
                # 總獎勵
                total_turn_reward = immediate_reward + terminal_reward
                total_reward += total_turn_reward
                
                # 創建最終記錄
                final_record = {
                    'turn': turn_data['turn'],
                    'state': turn_data['state'],
                    'action': turn_data['action'],
                    'reward': total_turn_reward,
                    'next_state': turn_data['next_state'],
                    
                    # 詳細獎勵資訊
                    'reward_breakdown': {
                        'immediate_reward': immediate_reward,
                        'terminal_reward': terminal_reward,
                        'total_reward': total_turn_reward,
                        **immediate_breakdown,
                        **terminal_breakdown
                    },
                    
                    # 原始資訊（用於分析）
                    'reward_requirements': reward_req,
                    'script_info': turn_data.get('script_info', {}),
                    'dialogue_info': turn_data.get('dialogue_info', {}),
                    
                    # 計算細節
                    'calculation_details': {
                        'slot_filled': slot_filled,
                        'deviation_level': deviation_level,
                        'distance_improvement': distance_improvement,
                        'is_terminal': is_terminal,
                        'r_case_value': self.get_R_case(slot_filled, deviation_level)
                    }
                }
                
                final_data.append(final_record)
                
                # 每10輪輸出進度
                if (i + 1) % 10 == 0:
                    logger.info(f"已處理 {i + 1}/{len(rl_data)} 輪，當前總獎勵: {total_reward:.3f}")
                
            except Exception as e:
                logger.error(f"處理第 {i+1} 輪資料時發生錯誤: {e}")
                # 如果出錯，創建一個零獎勵記錄
                final_record = {
                    'turn': turn_data.get('turn', i+1),
                    'state': turn_data.get('state', [0,0,0,0,0]),
                    'action': turn_data.get('action', [0]*12),
                    'reward': 0.0,
                    'next_state': turn_data.get('next_state', [0,0,0,0,0]),
                    'reward_breakdown': {'error': str(e)},
                    'calculation_details': {'error': True}
                }
                final_data.append(final_record)
        
        logger.info(f"獎勵計算完成！總獎勵: {total_reward:.3f}")
        return final_data

    def analyze_rewards(self, final_data: List[Dict]) -> Dict:
        """分析獎勵分布和統計"""
        if not final_data:
            return {}
        
        try:
            rewards = [record['reward'] for record in final_data]
            slot_rewards = [record['reward_breakdown'].get('slot_reward', 0) for record in final_data]
            deviation_rewards = [record['reward_breakdown'].get('deviation_reward', 0) for record in final_data]
            distance_rewards = [record['reward_breakdown'].get('distance_reward', 0) for record in final_data]
            
            # 基本統計
            stats = {
                'total_turns': len(final_data),
                'total_reward': sum(rewards),
                'average_reward': np.mean(rewards),
                'max_reward': max(rewards),
                'min_reward': min(rewards),
                'std_reward': np.std(rewards),
                
                'slot_rewards': {
                    'total': sum(slot_rewards),
                    'average': np.mean(slot_rewards),
                    'percentage': sum(slot_rewards) / sum(rewards) if sum(rewards) != 0 else 0
                },
                
                'deviation_rewards': {
                    'total': sum(deviation_rewards),
                    'average': np.mean(deviation_rewards),
                    'percentage': sum(deviation_rewards) / sum(rewards) if sum(rewards) != 0 else 0
                },
                
                'distance_rewards': {
                    'total': sum(distance_rewards),
                    'average': np.mean(distance_rewards),
                    'percentage': sum(distance_rewards) / sum(rewards) if sum(rewards) != 0 else 0
                }
            }
            
            # 偏離度統計
            deviation_stats = {}
            for level in range(4):
                level_records = [r for r in final_data 
                               if r['calculation_details'].get('deviation_level') == level]
                if level_records:
                    level_rewards = [r['reward'] for r in level_records]
                    deviation_stats[level] = {
                        'count': len(level_records),
                        'total_reward': sum(level_rewards),
                        'average_reward': np.mean(level_rewards)
                    }
                else:
                    deviation_stats[level] = {'count': 0, 'total_reward': 0, 'average_reward': 0}
            
            stats['deviation_statistics'] = deviation_stats
            
            # 填槽統計
            slot_filled_count = sum(1 for r in final_data 
                                  if r['calculation_details'].get('slot_filled', 0) == 1)
            stats['slot_filling'] = {
                'success_count': slot_filled_count,
                'success_rate': slot_filled_count / len(final_data),
                'failure_count': len(final_data) - slot_filled_count
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"分析獎勵時發生錯誤: {e}")
            return {}

def load_rl_data(file_path: str) -> List[Dict]:
    """載入RL資料檔案"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'state_action_data' in data:
            logger.info(f"成功載入 RL 資料，共 {len(data['state_action_data'])} 輪對話")
            return data['state_action_data']
        else:
            logger.error("RL 資料格式錯誤，找不到 'state_action_data' 鍵")
            return []
            
    except Exception as e:
        logger.error(f"載入 RL 資料時發生錯誤: {e}")
        return []

def save_final_data(final_data: List[Dict], stats: Dict, output_path: str) -> str:
    """保存最終的(s,a,r,s')資料"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"{output_path}/final_rl_data_{timestamp}.json"
    
    try:
        final_output = {
            'timestamp': timestamp,
            'reward_parameters': {
                'alpha': 15.0,
                'phi': 5.0,
                'theta': 3.0,
                'R_goal': 300.0,
                'mu': 30.0,
                'T_safe': 8,
                'delta': 10.0
            },
            'statistics': stats,
            'data': final_data
        }
        
        os.makedirs(output_path, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
        
        logger.info(f"最終資料已保存至: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"保存最終資料時發生錯誤: {e}")
        return ""

def main():
    """主函數"""
    import os
    
    # 輸入和輸出路徑
    rl_data_file = "outputs/dialogues/rl_data_20250721_161912.json"  # 請替換為實際路徑
    output_path = "outputs/final_rl_data"
    
    # 檢查輸入檔案是否存在
    if not os.path.exists(rl_data_file):
        logger.error(f"找不到 RL 資料檔案: {rl_data_file}")
        return
    
    # 載入 RL 資料
    rl_data = load_rl_data(rl_data_file)
    if not rl_data:
        logger.error("沒有可用的 RL 資料")
        return
    
    # 初始化獎勵計算器（使用新的參數）
    reward_params = RewardParameters(
        alpha=15.0,
        phi=5.0,
        theta=3.0,
        R_goal=300.0,
        mu=30.0,
        T_safe=100,
        delta=10.0
    )
    
    calculator = RewardCalculator(reward_params)
    
    # 計算所有獎勵
    logger.info("開始計算獎勵...")
    final_data = calculator.calculate_all_rewards(rl_data)
    
    # 分析獎勵
    logger.info("分析獎勵統計...")
    stats = calculator.analyze_rewards(final_data)
    
    # 保存最終資料
    output_file = save_final_data(final_data, stats, output_path)
    
    # 輸出統計摘要
    print(f"\n=== 獎勵計算完成！ ===")
    print(f"總對話輪數: {stats.get('total_turns', 0)}")
    print(f"總獎勵: {stats.get('total_reward', 0):.3f}")
    print(f"平均獎勵: {stats.get('average_reward', 0):.3f}")
    print(f"槽位填充成功率: {stats.get('slot_filling', {}).get('success_rate', 0):.1%}")
    
    # 顯示獎勵組成
    if 'slot_rewards' in stats:
        print(f"\n獎勵組成:")
        print(f"  槽位獎勵: {stats['slot_rewards']['total']:.3f} ({stats['slot_rewards']['percentage']:.1%})")
        print(f"  偏離獎勵: {stats['deviation_rewards']['total']:.3f} ({stats['deviation_rewards']['percentage']:.1%})")
        print(f"  距離獎勵: {stats['distance_rewards']['total']:.3f} ({stats['distance_rewards']['percentage']:.1%})")
    
    # 顯示偏離度統計
    if 'deviation_statistics' in stats:
        print(f"\n偏離度統計:")
        for level in range(4):
            level_stats = stats['deviation_statistics'][level]
            print(f"  偏離度{level}: {level_stats['count']} 次，平均獎勵: {level_stats['average_reward']:+.3f}")
    
    print(f"\n最終資料已保存至: {output_file}")

if __name__ == "__main__":
    main()