"""
整合DQN訓練和視覺化的完整流程
訓練完成後自動生成視覺化結果
"""

import os
import json
import jsonlines
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

# 導入核心模組
from dueling_dqn import DuelingDQNAgent
from data_preprocessor import DialogueDataPreprocessor

from visualize_matrices import MatrixVisualizer
from f1_evaluator import F1Evaluator

# 設置日誌
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegratedDQNTrainer:
    """整合DQN訓練和視覺化的完整流程"""
    
    def __init__(self, output_dir: str = "outputs/"):
        self.output_dir = output_dir
        self.models_dir = os.path.join(output_dir, "models")
        self.matrices_dir = os.path.join(output_dir, "matrices")
        self.visualizations_dir = os.path.join(output_dir, "visualizations")
        
        # 創建輸出目錄
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.matrices_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
        
        # 初始化組件
        self.agent = DuelingDQNAgent(
            state_dim=5,
            action_dim=12,
            learning_rate=0.001,  # 提高學習率（從0.0005提高）
            gamma=0.99,  # 恢復較高的折扣因子
            epsilon=1.0,
            epsilon_min=0.15,        # 從 0.1 提高到 0.15（保持更多探索）
            epsilon_decay=0.9998,    # 從 0.9995 改為 0.9998（更慢的衰減）
            target_update=200,      # 從 100 增加到 200（更穩定的目標網路）
            memory_capacity=100000,  # 從 50000 增加到 100000
            batch_size=128,         # 從 64 增加到 128
            weight_decay=5e-5,  # 減少正則化（從1e-4降低）
            # 暫時關閉類別權重，先讓模型正常學習
            use_weighted_loss=False,  # 關閉加權損失
            positive_weight=1.0,  # 恢復平衡權重
            negative_weight=1.0  # 恢復平衡權重
        )
        
        self.visualizer = MatrixVisualizer(self.visualizations_dir)
        self.f1_evaluator = F1Evaluator()
        
        logger.info("整合DQN訓練器初始化完成")
    
    def run_complete_pipeline(self, 
                            dialogue_files: List[str],
                            epochs: int = 1000,  # 從500增加到1000
                            batch_size: int = 64,  # 增加批次大小
                            save_interval: int = 100,  # 每100輪保存一次
                            visualize_during_training: bool = True) -> Dict[str, Any]:
        """執行完整的訓練和視覺化流程"""
        
        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ========== 第1步：數據預處理 ==========
        logger.info("🔄 第1步：數據預處理")
        preprocessor = DialogueDataPreprocessor()
        all_records, all_statistics = preprocessor.process_multiple_files(dialogue_files)
        
        if not all_records:
            logger.error("沒有有效的訓練數據")
            return results
        
        # 創建訓練/測試分割
        train_records, test_records = preprocessor.create_train_test_split(
            all_records, test_ratio=0.2
        )
        
        logger.info(f"數據分割完成: 訓練集 {len(train_records)} 條, 測試集 {len(test_records)} 條")
        
        # ========== 第2步：DQN訓練 ==========
        logger.info("🤖 第2步：DQN訓練")
        training_results = self.train_dqn(
            train_records=train_records,
            test_records=test_records,
            epochs=epochs,
            batch_size=batch_size,
            save_interval=save_interval,
            visualize_during_training=visualize_during_training
        )
        
        results['training'] = training_results
        
        # ========== 生成訓練曲線 ==========
        logger.info("📈 生成訓練曲線")
        curves_path = self.plot_training_curves(training_results, timestamp)
        results['training_curves'] = curves_path
        
        # ========== 第3步：提取Q矩陣 ==========
        logger.info("📈 第3步：提取Q矩陣")
        q_matrix = self.extract_q_matrix(training_results['final_model_path'])
        results['q_matrix'] = q_matrix
        
        # ========== 第4步：F1分數評估 ==========
        logger.info("🎯 第4步：F1分數評估")
        f1_results = self.evaluate_f1_score(test_records, training_results['final_model_path'])
        results['f1_evaluation'] = f1_results
        
        # 保存F1評估結果
        f1_file_path = self.save_f1_results(f1_results, timestamp)
        results['f1_file_path'] = f1_file_path
        results['f1_viz_path'] = os.path.join(self.visualizations_dir, f"f1_evaluation_{timestamp}.png")
        
        # ========== 第5步：提取R矩陣統計 ==========
        logger.info("📊 第5步：提取R矩陣統計")
        r_matrix_stats = self.extract_r_matrix_stats(all_records)
        results['r_matrix_stats'] = r_matrix_stats
        
        # ========== 第6步：保存模型權重詳細信息 ==========
        logger.info("💾 第6步：保存模型權重詳細信息")
        weights_file_path = self.save_model_weights_info(training_results['final_model_path'], timestamp)
        results['weights_file_path'] = weights_file_path
        
        # ========== 第7步：保存R矩陣和Q矩陣數值 ==========
        logger.info("🔢 第7步：保存R矩陣和Q矩陣數值")
        r_matrix = self._rebuild_r_matrix_for_visualization(r_matrix_stats)
        matrices_file_path = self.save_matrices_numerical_data(r_matrix, q_matrix, r_matrix_stats, timestamp)
        results['matrices_file_path'] = matrices_file_path
        
        # ========== 第8步：生成視覺化 ==========
        logger.info("🎨 第8步：生成視覺化")
        visualization_results = self.generate_visualizations(q_matrix, r_matrix_stats)
        results['visualizations'] = visualization_results
        
        # ========== 第9步：生成總結報告 ==========
        logger.info("📝 第9步：生成總結報告")
        report_path = self.generate_summary_report(results, timestamp)
        results['report_path'] = report_path
        
        logger.info("🎉 完整流程執行完成！")
        return results
    
    def train_dqn(self, 
                  train_records: List[Dict[str, Any]],
                  test_records: List[Dict[str, Any]],
                  epochs: int,
                  batch_size: int,
                  save_interval: int,
                  visualize_during_training: bool) -> Dict[str, Any]:
        """DQN訓練流程 - 加入早停機制和驗證監控"""
        
        # 分析資料平衡情況
        positive_rewards = sum(1 for r in train_records if r['reward'] > 0)
        negative_rewards = sum(1 for r in train_records if r['reward'] <= 0)
        logger.info(f"訓練資料平衡: 正獎勵 {positive_rewards}, 負獎勵 {negative_rewards}")
        
        # 如果資料極度不平衡，進行平衡調整
        if positive_rewards == 0 or negative_rewards == 0:
            logger.warning("資料極度不平衡，可能導致過擬合")
            
        # 將訓練數據加入經驗回放緩衝區
        for record in train_records:
            self.agent.memory.push(
                record['state'],
                record['action'],
                record['reward'],
                record['next_state'],
                record['done']
            )
        
        logger.info(f"已將 {len(train_records)} 條記錄加入經驗回放緩衝區")
        
        # 訓練指標
        losses = []
        train_rewards = []
        test_accuracies = []
        
        # 新增：詳細指標記錄
        f1_scores = []
        precisions = []
        recalls = []
        action_accuracies = []
        epsilon_values = []
        
        # 早停機制
        best_test_accuracy = 0.0
        best_f1_score = 0.0
        patience_counter = 0
        patience = 100  # 從50增加到100，給模型更多學習機會
        
        # 開始訓練
        for epoch in range(epochs):
            # 訓練一個批次
            loss = self.agent.train(batch_size)
            
            # 如果觸發早停機制（損失為inf）
            if loss == float('inf'):
                logger.warning(f"在第 {epoch + 1} 輪觸發早停機制")
                break
                
            losses.append(loss)
            
            # 計算訓練獎勵
            epoch_rewards = [record['reward'] for record in train_records]
            train_rewards.append(np.mean(epoch_rewards))
            
            # 定期評估
            if (epoch + 1) % save_interval == 0:
                # 在測試集上評估
                test_accuracy = self.evaluate_on_test_data(test_records)
                test_accuracies.append(test_accuracy)
                
                # 評估F1分數等詳細指標
                f1_results = self.evaluate_f1_score(test_records, os.path.join(self.models_dir, f"dqn_model_epoch_{epoch + 1}.pth"))
                f1_scores.append(f1_results.get('f1_score', 0))
                precisions.append(f1_results.get('precision', 0))
                recalls.append(f1_results.get('recall', 0))
                action_accuracies.append(f1_results.get('action_accuracy', 0))
                epsilon_values.append(self.agent.epsilon)
                
                # 早停機制：基於F1分數而不是準確率
                current_f1 = f1_results.get('f1_score', 0)
                if current_f1 > best_f1_score:
                    best_f1_score = current_f1
                    patience_counter = 0
                    
                    # 保存最佳模型
                    best_model_path = os.path.join(self.models_dir, f"best_dqn_model_epoch_{epoch + 1}.pth")
                    self.agent.save_model(best_model_path)
                    logger.info(f"💾 保存最佳模型: {best_model_path} (F1: {best_f1_score:.3f})")
                else:
                    patience_counter += 1
                    
                # 如果連續多輪沒有改善，提前停止
                if patience_counter >= patience:
                    logger.warning(f"測試準確率連續{patience}輪未改善，提前停止訓練")
                    logger.info(f"最佳測試準確率: {best_test_accuracy:.2%}")
                    break
                
                # 保存中間模型
                model_path = os.path.join(self.models_dir, f"dqn_model_epoch_{epoch + 1}.pth")
                self.agent.save_model(model_path)
                
                # 可選：中間視覺化
                if visualize_during_training and (epoch + 1) % (save_interval * 2) == 0:
                    logger.info(f"🎨 生成第 {epoch + 1} 輪的中間視覺化")
                    q_matrix = self.extract_q_matrix(model_path)
                    self.visualizer.visualize_q_matrix(q_matrix)
                
                logger.info(f"輪數 {epoch + 1}/{epochs} - "
                           f"損失: {loss:.4f}, "
                           f"測試準確率: {test_accuracy:.2%}, "
                           f"最佳準確率: {best_test_accuracy:.2%}, "
                           f"Epsilon: {self.agent.epsilon:.4f}, "
                           f"耐心計數: {patience_counter}/{patience}")
                
                # 每100輪顯示詳細統計
                if (epoch + 1) % 100 == 0:
                    logger.info(f"📊 第 {epoch + 1} 輪詳細統計:")
                    logger.info(f"   平均損失: {np.mean(losses[-100:]):.4f}")
                    logger.info(f"   損失標準差: {np.std(losses[-100:]):.4f}")
                    logger.info(f"   測試準確率趨勢: {np.mean(test_accuracies[-5:]) if len(test_accuracies) >= 5 else test_accuracy:.2%}")
        
        # 保存最終模型
        final_model_path = os.path.join(self.models_dir, "final_dqn_model.pth")
        self.agent.save_model(final_model_path)
        
        return {
            'losses': losses,
            'train_rewards': train_rewards,
            'test_accuracies': test_accuracies,
            'f1_scores': f1_scores,
            'precisions': precisions,
            'recalls': recalls,
            'action_accuracies': action_accuracies,
            'epsilon_values': epsilon_values,
            'final_model_path': final_model_path,
            'best_test_accuracy': best_test_accuracy,
            'best_f1_score': best_f1_score,
            'training_stopped_early': patience_counter >= patience,
            'total_epochs_trained': epoch + 1
        }
    
    def extract_q_matrix(self, model_path: str) -> np.ndarray:
        """從訓練好的模型提取Q矩陣"""
        return self.visualizer.extract_q_matrix(model_path)
    
    def evaluate_on_test_data(self, test_records: List[Dict[str, Any]]) -> float:
        """在測試數據上評估模型"""
        if not test_records:
            return 0.0
        
        correct_actions = 0
        
        for record in test_records:
            state = record['state']
            expected_action = record['action']
            
            # 使用訓練好的模型預測動作
            predicted_action = self.agent.select_action(state, training=False)
            
            # 檢查動作是否正確
            expected_idx = np.argmax(expected_action)
            predicted_idx = np.argmax(predicted_action)
            
            if expected_idx == predicted_idx:
                correct_actions += 1
        
        return correct_actions / len(test_records)
    
    def evaluate_f1_score(self, test_records: List[Dict[str, Any]], model_path: str) -> Dict[str, float]:
        """評估F1分數"""
        if not test_records:
            logger.warning("沒有測試數據，跳過F1評估")
            return {}
        
        # 載入訓練好的模型
        self.agent.load_model(model_path)
        
        # 使用F1評估器進行評估
        f1_results = self.f1_evaluator.evaluate_slot_filling_f1(test_records, self.agent)
        
        logger.info("F1分數評估完成:")
        logger.info(f"  F1分數: {f1_results.get('f1_score', 0):.3f}")
        logger.info(f"  精確率: {f1_results.get('precision', 0):.3f}")
        logger.info(f"  召回率: {f1_results.get('recall', 0):.3f}")
        logger.info(f"  動作準確率: {f1_results.get('action_accuracy', 0):.3f}")
        logger.info(f"  槽位填充準確率: {f1_results.get('slot_filling_accuracy', 0):.3f}")
        
        return f1_results
    
    def extract_r_matrix_stats(self, all_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """從對話記錄中提取R矩陣統計信息（不重新計算，只統計）"""
        logger.info("📊 從對話記錄中提取R矩陣統計...")
        
        # 統計獎勵分佈
        rewards = [record['reward'] for record in all_records]
        
        # 統計狀態-動作對的獎勵
        state_action_rewards = {}
        for record in all_records:
            state = tuple(record['state'])
            action = record['action']
            reward = record['reward']
            
            # 將動作轉換為索引
            if isinstance(action, list):
                action_idx = action.index(1) if 1 in action else 0
            else:
                action_idx = int(action)
            
            key = (state, action_idx)
            if key not in state_action_rewards:
                state_action_rewards[key] = []
            state_action_rewards[key].append(reward)
        
        # 計算平均獎勵
        avg_rewards = {}
        for key, reward_list in state_action_rewards.items():
            avg_rewards[key] = np.mean(reward_list)
        
        # 統計信息
        stats = {
            'total_records': len(all_records),
            'unique_state_action_pairs': len(state_action_rewards),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'median_reward': np.median(rewards),
            'positive_rewards': np.sum(np.array(rewards) > 0),
            'negative_rewards': np.sum(np.array(rewards) < 0),
            'zero_rewards': np.sum(np.array(rewards) == 0),
            'reward_distribution': rewards,
            # 將tuple key轉換為字符串
            'state_action_avg_rewards': {str(k): v for k, v in avg_rewards.items()}
        }
        
        logger.info(f"R矩陣統計完成:")
        logger.info(f"  總記錄數: {stats['total_records']}")
        logger.info(f"  唯一狀態-動作對: {stats['unique_state_action_pairs']}")
        logger.info(f"  平均獎勵: {stats['mean_reward']:.3f}")
        logger.info(f"  獎勵範圍: [{stats['min_reward']:.3f}, {stats['max_reward']:.3f}]")
        
        return stats
    
    def generate_visualizations(self, 
                              q_matrix: np.ndarray, 
                              r_matrix_stats: Dict) -> Dict[str, str]:
        """生成視覺化圖表"""
        
        # 重建R矩陣用於可視化
        r_matrix = self._rebuild_r_matrix_for_visualization(r_matrix_stats)
        
        # 生成Q矩陣視覺化
        q_viz_path = self.visualizer.visualize_q_matrix(q_matrix)
        
        # 生成R矩陣視覺化
        r_stats_for_viz = {
            'total_records': r_matrix_stats['total_records'],
            'non_zero_entries': r_matrix_stats['unique_state_action_pairs'],
            'mean_reward': r_matrix_stats['mean_reward'],
            'std_reward': r_matrix_stats['std_reward'],
            'min_reward': r_matrix_stats['min_reward'],
            'max_reward': r_matrix_stats['max_reward']
        }
        r_viz_path = self.visualizer.visualize_r_matrix(r_matrix, r_stats_for_viz)
        
        # 生成R vs Q比較視覺化
        comparison_path = self.visualizer.compare_r_q_matrices(r_matrix, q_matrix)
        
        # 保存矩陣數據
        data_path = self.visualizer.save_matrices_data(r_matrix, q_matrix, r_stats_for_viz)
        
        return {
            'r_matrix_viz': r_viz_path,
            'q_matrix_viz': q_viz_path,
            'comparison_viz': comparison_path,
            'data_file': data_path
        }
    
    def _rebuild_r_matrix_for_visualization(self, r_matrix_stats: Dict) -> np.ndarray:
        """從統計信息重建R矩陣用於可視化"""
        r_matrix = np.zeros((32, 12))
        
        # 狀態轉換為索引的輔助函數
        def state_to_index(state):
            if len(state) != 5:
                return 0
            index = 0
            for i, val in enumerate(state):
                if val > 0:
                    index += 2 ** (4 - i)
            return min(index, 31)
        
        # 填充R矩陣
        for (state, action_idx), avg_reward in r_matrix_stats['state_action_avg_rewards'].items():
            state_idx = state_to_index(list(state))
            if 0 <= state_idx < 32 and 0 <= action_idx < 12:
                r_matrix[state_idx, action_idx] = avg_reward
        
        return r_matrix
    
    def save_f1_results(self, f1_results: Dict[str, float], timestamp: str) -> str:
        """保存F1評估結果到文件"""
        
        # 生成F1評估報告
        f1_report = self.f1_evaluator.generate_f1_report(f1_results)
        
        # 保存詳細的F1結果
        f1_data = {
            'timestamp': timestamp,
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'f1_evaluation': f1_results,
            'detailed_report': f1_report
        }
        
        f1_file_path = os.path.join(self.output_dir, f"f1_evaluation_{timestamp}.json")
        
        with open(f1_file_path, 'w', encoding='utf-8') as f:
            json.dump(f1_data, f, ensure_ascii=False, indent=2)
        
        # 也保存為可讀的markdown格式
        f1_md_path = os.path.join(self.output_dir, f"f1_evaluation_{timestamp}.md")
        with open(f1_md_path, 'w', encoding='utf-8') as f:
            f.write(f1_report)
        
        # 🆕 生成F1評估視覺化圖表
        f1_viz_path = self.generate_f1_visualization(f1_results, timestamp)
        
        logger.info(f"F1評估結果已保存:")
        logger.info(f"  JSON格式: {os.path.basename(f1_file_path)}")
        logger.info(f"  Markdown格式: {os.path.basename(f1_md_path)}")
        logger.info(f"  視覺化圖表: {os.path.basename(f1_viz_path)}")
        
        return f1_file_path

    def generate_f1_visualization(self, f1_results: Dict[str, float], timestamp: str) -> str:
        """生成F1評估的視覺化圖表"""
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        # 設置中文字體
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 創建圖表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('F1分數評估結果視覺化', fontsize=16, fontweight='bold')
        
        # 1. 混淆矩陣
        ax1 = axes[0, 0]
        
        # 計算混淆矩陣的值
        tp = f1_results.get('true_positives', 0)
        fp = f1_results.get('false_positives', 0)
        fn = f1_results.get('false_negatives', 0)
        total_samples = f1_results.get('total_test_samples', 0)
        tn = total_samples - tp - fp - fn
        
        # 創建混淆矩陣
        confusion_matrix = np.array([[tn, fp], [fn, tp]])
        
        # 繪製混淆矩陣熱力圖
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['預測: 0 (不填充)', '預測: 1 (填充)'],
                   yticklabels=['實際: 0 (不填充)', '實際: 1 (填充)'],
                   ax=ax1, cbar_kws={'label': '樣本數量'})
        
        ax1.set_title('混淆矩陣')
        ax1.set_xlabel('預測結果')
        ax1.set_ylabel('實際結果')
        
        # 添加標籤說明
        ax1.text(0.5, 0.1, f'TN = {tn}', ha='center', va='center', 
                transform=ax1.transData, fontsize=12, fontweight='bold', color='white')
        ax1.text(1.5, 0.1, f'FP = {fp}', ha='center', va='center',
                transform=ax1.transData, fontsize=12, fontweight='bold', color='white')
        ax1.text(0.5, 0.9, f'FN = {fn}', ha='center', va='center',
                transform=ax1.transData, fontsize=12, fontweight='bold', color='white')
        ax1.text(1.5, 0.9, f'TP = {tp}', ha='center', va='center',
                transform=ax1.transData, fontsize=12, fontweight='bold', color='white')
        
        # 2. F1指標柱狀圖
        ax2 = axes[0, 1]
        
        metrics = ['精確率\n(Precision)', '召回率\n(Recall)', 'F1分數\n(F1-Score)', '動作準確率\n(Action Acc)']
        values = [
            f1_results.get('precision', 0),
            f1_results.get('recall', 0),
            f1_results.get('f1_score', 0),
            f1_results.get('action_accuracy', 0)
        ]
        
        colors = ['skyblue', 'lightgreen', 'gold', 'lightcoral']
        bars = ax2.bar(metrics, values, color=colors, alpha=0.8)
        
        # 添加數值標籤
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_title('F1評估指標')
        ax2.set_ylabel('分數')
        ax2.set_ylim(0, 1.1)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. 混淆矩陣組成餅圖
        ax3 = axes[1, 0]
        
        labels = ['真負例 (TN)', '假正例 (FP)', '假負例 (FN)', '真正例 (TP)']
        sizes = [tn, fp, fn, tp]
        colors_pie = ['lightblue', 'lightcoral', 'lightsalmon', 'lightgreen']
        explode = (0.05, 0.05, 0.05, 0.1)  # 突出TP
        
        # 過濾掉值為0的項目
        non_zero_data = [(label, size, color, exp) for label, size, color, exp in 
                        zip(labels, sizes, colors_pie, explode) if size > 0]
        
        if non_zero_data:
            labels_nz, sizes_nz, colors_nz, explode_nz = zip(*non_zero_data)
            
            wedges, texts, autotexts = ax3.pie(sizes_nz, labels=labels_nz, colors=colors_nz,
                                              autopct='%1.1f%%', startangle=90, explode=explode_nz)
            
            # 美化文字
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        ax3.set_title('測試樣本分佈')
        
        # 4. 詳細統計信息
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # 計算額外統計
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 等於召回率
        accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
        
        stats_text = f"""
F1評估詳細統計:

混淆矩陣數值:
• 真正例 (TP): {tp}
• 假正例 (FP): {fp}
• 假負例 (FN): {fn}
• 真負例 (TN): {tn}
• 總樣本數: {total_samples}

評估指標:
• F1分數: {f1_results.get('f1_score', 0):.3f}
• 精確率: {f1_results.get('precision', 0):.3f}
• 召回率: {f1_results.get('recall', 0):.3f}
• 特異性: {specificity:.3f}
• 準確率: {accuracy:.3f}
• 動作準確率: {f1_results.get('action_accuracy', 0):.3f}
• 槽位填充準確率: {f1_results.get('slot_filling_accuracy', 0):.3f}

指標說明:
• F1分數: 精確率和召回率的調和平均
• 精確率: TP/(TP+FP) - 預測正例的準確性
• 召回率: TP/(TP+FN) - 找出正例的能力
• 特異性: TN/(TN+FP) - 找出負例的能力
• 準確率: (TP+TN)/Total - 整體預測準確性
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        # 保存圖表
        f1_viz_path = os.path.join(self.visualizations_dir, f"f1_evaluation_{timestamp}.png")
        
        try:
            plt.savefig(f1_viz_path, dpi=300, bbox_inches='tight')
            logger.info(f"F1評估視覺化已保存: {os.path.basename(f1_viz_path)}")
        except Exception as e:
            logger.error(f"保存F1視覺化時發生錯誤: {e}")
        
        # 關閉圖表避免阻塞
        plt.close()
        
        return f1_viz_path

    def save_model_weights_info(self, model_path: str, timestamp: str) -> str:
        """保存模型權重詳細信息"""
        
        import torch
        
        # 載入模型檢查點
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 提取權重信息
        weights_info = {
            'timestamp': timestamp,
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': model_path,
            'hyperparameters': checkpoint.get('hyperparameters', {}),
            'training_state': {
                'epsilon': checkpoint.get('epsilon', 0),
                'update_count': checkpoint.get('update_count', 0),
                'total_steps': checkpoint.get('total_steps', 0)
            },
            'training_metrics': {
                'training_rewards': checkpoint.get('training_rewards', []),
                'training_losses': checkpoint.get('training_losses', []),
                'final_loss': checkpoint.get('training_losses', [0])[-1] if checkpoint.get('training_losses') else 0
            },
            'network_architecture': {
                'q_network_layers': [],
                'target_network_layers': []
            }
        }
        
        # 分析網路結構
        q_network_state = checkpoint.get('q_network_state_dict', {})
        for layer_name, weights in q_network_state.items():
            if hasattr(weights, 'shape'):
                # 將權重轉換為浮點數以計算統計
                weights_float = weights.float() if weights.dtype in [torch.long, torch.int, torch.int32, torch.int64] else weights
                
                weights_info['network_architecture']['q_network_layers'].append({
                    'layer_name': layer_name,
                    'shape': list(weights.shape),
                    'total_params': weights.numel(),
                    'dtype': str(weights.dtype),
                    'weight_stats': {
                        'mean': float(weights_float.mean()),
                        'std': float(weights_float.std()),
                        'min': float(weights_float.min()),
                        'max': float(weights_float.max())
                    }
                })
        
        # 計算總參數數量
        total_params = sum(layer['total_params'] for layer in weights_info['network_architecture']['q_network_layers'])
        weights_info['network_architecture']['total_parameters'] = total_params
        
        # 保存權重信息
        weights_file_path = os.path.join(self.models_dir, f"model_weights_info_{timestamp}.json")
        
        with open(weights_file_path, 'w', encoding='utf-8') as f:
            json.dump(weights_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"模型權重信息已保存: {os.path.basename(weights_file_path)}")
        logger.info(f"  總參數數量: {total_params:,}")
        logger.info(f"  網路層數: {len(weights_info['network_architecture']['q_network_layers'])}")
        
        return weights_file_path

    def save_matrices_numerical_data(self, r_matrix: np.ndarray, q_matrix: np.ndarray, 
                                   r_matrix_stats: Dict, timestamp: str) -> str:
        """保存R矩陣和Q矩陣的完整數值數據"""
        logger.info("🔢 第7步：保存R矩陣和Q矩陣數值")
        
        # 確保所有數值都是Python原生類型
        def convert_to_native(obj):
            """遞歸轉換所有NumPy類型為Python原生類型"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj
        
        # 轉換矩陣為列表格式
        r_matrix_list = r_matrix.tolist()
        q_matrix_list = q_matrix.tolist()
        
        # 準備數據
        matrices_data = {
            'timestamp': timestamp,
            'r_matrix': {
                'data': r_matrix_list,
                'shape': [int(d) for d in r_matrix.shape],
                'statistics': convert_to_native(r_matrix_stats)
            },
            'q_matrix': {
                'data': q_matrix_list,
                'shape': [int(d) for d in q_matrix.shape],
                'statistics': {
                    'mean': float(np.mean(q_matrix)),
                    'std': float(np.std(q_matrix)),
                    'min': float(np.min(q_matrix)),
                    'max': float(np.max(q_matrix)),
                    'total_elements': int(q_matrix.size)
                }
            },
            'state_labels': [f"[{','.join(map(str, self._index_to_state(i)))}]" for i in range(32)],
            'action_labels': [
                "背景→用藥", "背景→睡眠", "背景→活動", "背景→飲食",
                "作息→用藥", "作息→睡眠", "作息→活動", "作息→飲食",
                "興趣→用藥", "興趣→睡眠", "興趣→活動", "興趣→飲食"
            ],
            'comparison_analysis': {
                'r_q_correlation': float(np.corrcoef(r_matrix.flatten(), q_matrix.flatten())[0, 1]),
                'common_max_actions': self._analyze_common_max_actions(r_matrix, q_matrix)
            }
        }
        
        # 保存為JSON格式
        matrices_file_path = os.path.join(self.matrices_dir, f"matrices_numerical_data_{timestamp}.json")
        
        with open(matrices_file_path, 'w', encoding='utf-8') as f:
            json.dump(matrices_data, f, ensure_ascii=False, indent=2)
        
        # 也保存為NumPy格式以便後續分析
        r_matrix_npy_path = os.path.join(self.matrices_dir, f"r_matrix_{timestamp}.npy")
        q_matrix_npy_path = os.path.join(self.matrices_dir, f"q_matrix_{timestamp}.npy")
        
        np.save(r_matrix_npy_path, r_matrix)
        np.save(q_matrix_npy_path, q_matrix)
        
        logger.info(f"矩陣數值數據已保存:")
        logger.info(f"  JSON格式: {os.path.basename(matrices_file_path)}")
        logger.info(f"  R矩陣NumPy: {os.path.basename(r_matrix_npy_path)}")
        logger.info(f"  Q矩陣NumPy: {os.path.basename(q_matrix_npy_path)}")
        
        return matrices_file_path

    def _index_to_state(self, index: int) -> List[int]:
        """將索引轉換為狀態向量"""
        if index < 0 or index >= 32:
            return [0, 0, 0, 0, 0]
        
        state = []
        for i in range(5):
            bit = (index >> (4 - i)) & 1
            state.append(bit)
        
        return state

    def _analyze_common_max_actions(self, r_matrix: np.ndarray, q_matrix: np.ndarray) -> Dict:
        """分析R矩陣和Q矩陣的最佳動作一致性"""
        
        r_max_actions = np.argmax(r_matrix, axis=1)
        q_max_actions = np.argmax(q_matrix, axis=1)
        
        # 計算一致性
        agreement = np.sum(r_max_actions == q_max_actions)
        agreement_rate = agreement / len(r_max_actions)
        
        # 找出不一致的狀態
        disagreement_states = np.where(r_max_actions != q_max_actions)[0]
        
        return {
            'agreement_count': int(agreement),
            'total_states': int(len(r_max_actions)),
            'agreement_rate': float(agreement_rate),
            'disagreement_states': [int(s) for s in disagreement_states],
            'r_preferred_actions': [int(a) for a in r_max_actions],
            'q_preferred_actions': [int(a) for a in q_max_actions]
        }

    def generate_summary_report(self, results: Dict[str, Any], timestamp: str) -> str:
        """生成總結報告"""
        report_content = f"""
# DQN訓練和視覺化完整報告
生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 訓練結果摘要
- 最終損失: {results['training']['losses'][-1]:.4f}
- 最終測試準確率: {results['training']['final_evaluation']:.2%}
- 訓練輪數: {results['training']['total_epochs_trained']}
- 訓練是否提前停止: {results['training']['training_stopped_early']}

## 2. R矩陣統計（從對話記錄提取）
- 總記錄數: {results['r_matrix_stats']['total_records']}
- 唯一狀態-動作對: {results['r_matrix_stats']['unique_state_action_pairs']}
- 平均獎勵: {results['r_matrix_stats']['mean_reward']:.3f}
- 獎勵範圍: [{results['r_matrix_stats']['min_reward']:.3f}, {results['r_matrix_stats']['max_reward']:.3f}]
- 正獎勵記錄: {results['r_matrix_stats']['positive_rewards']}
- 負獎勵記錄: {results['r_matrix_stats']['negative_rewards']}
- 零獎勵記錄: {results['r_matrix_stats']['zero_rewards']}

## 3. Q矩陣統計
- 矩陣形狀: {results['q_matrix'].shape}
- Q值範圍: [{results['q_matrix'].min():.3f}, {results['q_matrix'].max():.3f}]
- 平均Q值: {results['q_matrix'].mean():.3f}

## 4. F1分數評估
- F1分數: {results['f1_evaluation'].get('f1_score', 0):.3f}
- 精確率: {results['f1_evaluation'].get('precision', 0):.3f}
- 召回率: {results['f1_evaluation'].get('recall', 0):.3f}
- 動作準確率: {results['f1_evaluation'].get('action_accuracy', 0):.3f}
- 槽位填充準確率: {results['f1_evaluation'].get('slot_filling_accuracy', 0):.3f}
- 測試樣本數: {results['f1_evaluation'].get('total_test_samples', 0)}

## 5. 輸出文件
- 最終模型: {os.path.basename(results['training']['final_model_path'])}
- F1評估結果: {os.path.basename(results['f1_file_path'])}
- F1評估圖表: {os.path.basename(results['f1_viz_path'])}
- 模型權重信息: {os.path.basename(results['weights_file_path'])}
- 矩陣數值數據: {os.path.basename(results['matrices_file_path'])}
- R矩陣圖表: {os.path.basename(results['visualizations']['r_matrix_viz'])}
- Q矩陣圖表: {os.path.basename(results['visualizations']['q_matrix_viz'])}
- 比較圖表: {os.path.basename(results['visualizations']['comparison_viz'])}
- 視覺化數據: {os.path.basename(results['visualizations']['data_file'])}

## 6. 使用說明
1. **模型文件**: 可用於後續預測和繼續訓練
2. **F1評估**: 詳細的模型性能評估結果
3. **權重信息**: 模型參數和架構的詳細信息
4. **矩陣數值**: 完整的R矩陣和Q矩陣數值數據
5. **視覺化圖表**: 直觀的矩陣分析和比較
6. **所有數據**: 均已保存為JSON和NumPy格式，方便後續分析

## 7. 數據位置
- 模型文件: `{self.models_dir}/`
- 矩陣數據: `{self.matrices_dir}/`
- 視覺化: `{self.visualizations_dir}/`
- 報告文件: `{self.output_dir}/`
"""
        
        # 保存報告
        report_path = os.path.join(self.output_dir, f"training_report_{timestamp}.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"完整報告已保存: {os.path.basename(report_path)}")
        return report_path

    def plot_training_curves(self, metrics: Dict[str, List], timestamp: str) -> str:
        """繪製訓練曲線"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DQN訓練曲線', fontsize=16, fontproperties='Microsoft YaHei')
        
        # 1. 損失曲線
        ax1 = axes[0, 0]
        ax1.plot(metrics['losses'], 'b-', alpha=0.7)
        ax1.set_title('訓練損失', fontproperties='Microsoft YaHei')
        ax1.set_xlabel('訓練步數')
        ax1.set_ylabel('損失')
        ax1.grid(True, alpha=0.3)
        
        # 2. F1分數曲線
        ax2 = axes[0, 1]
        epochs = [i * 100 for i in range(len(metrics['f1_scores']))]
        ax2.plot(epochs, metrics['f1_scores'], 'g-', label='F1分數', linewidth=2)
        ax2.set_title('F1分數變化', fontproperties='Microsoft YaHei')
        ax2.set_xlabel('訓練輪數')
        ax2.set_ylabel('F1分數')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. 精確率vs召回率
        ax3 = axes[0, 2]
        ax3.plot(epochs, metrics['precisions'], 'r-', label='精確率', linewidth=2)
        ax3.plot(epochs, metrics['recalls'], 'b-', label='召回率', linewidth=2)
        ax3.set_title('精確率 vs 召回率', fontproperties='Microsoft YaHei')
        ax3.set_xlabel('訓練輪數')
        ax3.set_ylabel('分數')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. 動作準確率
        ax4 = axes[1, 0]
        ax4.plot(epochs, metrics['action_accuracies'], 'm-', linewidth=2)
        ax4.set_title('動作準確率', fontproperties='Microsoft YaHei')
        ax4.set_xlabel('訓練輪數')
        ax4.set_ylabel('準確率')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        # 5. Epsilon衰減曲線
        ax5 = axes[1, 1]
        ax5.plot(epochs, metrics['epsilon_values'], 'c-', linewidth=2)
        ax5.set_title('探索率(Epsilon)衰減', fontproperties='Microsoft YaHei')
        ax5.set_xlabel('訓練輪數')
        ax5.set_ylabel('Epsilon')
        ax5.set_ylim(0, 1.1)
        ax5.grid(True, alpha=0.3)
        
        # 6. 測試準確率
        ax6 = axes[1, 2]
        ax6.plot(epochs, metrics['test_accuracies'], 'orange', linewidth=2)
        ax6.set_title('測試準確率', fontproperties='Microsoft YaHei')
        ax6.set_xlabel('訓練輪數')
        ax6.set_ylabel('準確率')
        ax6.set_ylim(0, 1)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存圖片
        save_path = os.path.join(self.visualizations_dir, f"training_curves_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"訓練曲線已保存: {save_path}")
        return save_path

def main():
    """主函數 - 執行完整的整合流程"""
    
    # 設置對話文件路徑 - 使用所有可用的對話歷史檔案
    dialogue_files = [
        "outputs/dialogues/dialogue_history_20250713_192737.json",
        "outputs/dialogues/dialogue_history_20250713_201011.json",
        "outputs/dialogues/dialogue_history_20250713_210329.json",
        "outputs/dialogues/dialogue_history_20250713_220246.json",
        "outputs/dialogues/dialogue_history_20250713_221045.json",
        "outputs/dialogues/dialogue_history_20250713_224030.json",
        "outputs/dialogues/dialogue_history_20250713_234100.json",
        "outputs/dialogues/dialogue_history_20250714_000452.json",
        "outputs/dialogues/dialogue_history_20250714_111551.json",
        "outputs/dialogues/dialogue_history_20250714_113446.json",
        "outputs/dialogues/dialogue_history_20250714_115113.json",
        "outputs/dialogues/dialogue_history_20250714_120825.json",
        "outputs/dialogues/dialogue_history_20250714_140106.json",
        "outputs/dialogues/dialogue_history_20250714_140720.json",
        "outputs/dialogues/dialogue_history_20250714_141255.json",
        "outputs/dialogues/dialogue_history_20250714_145356.json",
        "outputs/dialogues/dialogue_history_20250714_154009.json",
        "outputs/dialogues/dialogue_history_20250714_160959.json",
        "outputs/dialogues/dialogue_history_20250714_162257.json",
        "outputs/dialogues/dialogue_history_20250714_172013.json"
    ]
    
    # 初始化整合訓練器
    trainer = IntegratedDQNTrainer()
    
    # 執行完整流程
    results = trainer.run_complete_pipeline(
        dialogue_files=dialogue_files,
        epochs=1000,  # 增加到1000輪
        batch_size=64,  # 增加批次大小
        save_interval=100,  # 每100輪保存一次
        visualize_during_training=True
    )
    
    # 輸出結果摘要
    print("\n" + "="*60)
    print("🎉 整合DQN訓練和視覺化完成！")
    print("="*60)
    print(f"📊 最終損失: {results['training']['losses'][-1]:.4f}")
    print(f"🎯 測試準確率: {results['training']['final_evaluation']:.2%}")
    print(f"🏆 F1分數: {results['f1_evaluation'].get('f1_score', 0):.3f}")
    print(f"📈 R矩陣統計: {results['r_matrix_stats']['unique_state_action_pairs']} 個狀態-動作對")
    print(f"💎 Q值範圍: [{results['q_matrix'].min():.3f}, {results['q_matrix'].max():.3f}]")
    print(f"📁 輸出目錄: {trainer.output_dir}")
    print(f"📝 完整報告: {os.path.basename(results['report_path'])}")
    print("="*60)

if __name__ == "__main__":
    main() 