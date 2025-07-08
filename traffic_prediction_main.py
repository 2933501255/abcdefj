import os
import cv2
import numpy as np
from rknn.api import RKNN
import time
from collections import Counter, deque
import threading
import queue
import signal
import sys
import json
from pypinyin import pinyin, Style
import subprocess
from edge_tts import Communicate
from playsound import playsound
import asyncio

# 导入LED控制模块
try:
    import gpiod
    LED_AVAILABLE = True
    print("✅ LED控制模块导入成功")
except ImportError:
    LED_AVAILABLE = False
    print("⚠️ LED控制模块不可用")

class TrafficPredictionMain:
    def __init__(self, model_path, camera_port=21):
        """
        交通预测主程序（独立进程）
        
        Args:
            model_path: RKNN模型文件路径
            camera_port: 摄像头端口号
        """
        self.model_path = model_path
        self.camera_port = camera_port
        self.rknn = None
        self.cap = None
        
        # LED控制初始化
        self.led_chip = None
        self.led_red = None
        self.led_green = None
        self.led_blue = None
        self.init_led()
        
        # 交通状态映射
        self.status_map = {0: '畅通', 1: '缓行', 2: '拥堵', 3: '封闭'}
        
        # LED颜色映射
        self.led_colors = {
            '畅通': (0, 1, 0),    # 绿色
            '缓行': (1, 1, 0),    # 黄色
            '拥堵': (1, 0, 0),    # 红色
            '封闭': (0, 0, 1),    # 蓝色
            '语音模式': (1, 0, 1), # 紫色
            '关闭': (0, 0, 0)     # 熄灭
        }
        
        # 预设语音提示词
        self.voice_prompts = {
            '畅通': "道路畅通，请安全驾驶，保持车距",
            '缓行': "前方道路缓行，建议减速慢行，注意保持安全距离",
            '拥堵': "前方道路拥堵，请耐心等待，避免频繁变道",
            '封闭': "前方道路封闭，请寻找替代路线"
        }
        
        # 预处理参数
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # 帧缓冲区
        self.frame_buffer = deque(maxlen=30)
        self.prediction_history = deque(maxlen=10)
        
        # 线程控制
        self.running = False
        self.prediction_paused = False
        self.frame_queue = queue.Queue(maxsize=10)
        
        # 语音状态管理
        self.last_announced_status = None
        self.status_change_count = 0
        self.min_status_duration = 3
        
        # 进程间通信文件
        self.status_file = '/tmp/traffic_status.json'
        self.command_file = '/tmp/traffic_command.txt'
        self.voice_trigger_file = '/tmp/voice_trigger.txt'
        
        # 清理旧的通信文件
        self.cleanup_communication_files()
        
        print(f"交通预测主程序初始化 - 模型: {model_path}, 摄像头端口: {camera_port}")
    
    def init_led(self):
        """初始化LED控制"""
        if not LED_AVAILABLE:
            print("⚠️ LED控制模块不可用，跳过LED初始化")
            return
        
        try:
            print("🔧 初始化LED控制...")
            
            # 检查gpiod模块是否正确加载
            print(f"🔍 gpiod模块版本: {gpiod.__version__ if hasattr(gpiod, '__version__') else '未知'}")
            
            # 获取可用的GPIO芯片
            try:
                available_chips = gpiod.ChipIter()
                chip_names = [chip.name for chip in available_chips]
                print(f"🔍 可用的GPIO芯片: {chip_names}")
            except Exception as e:
                print(f"⚠️ 无法获取GPIO芯片列表: {e}")
            
            # 初始化GPIO芯片
            try:
                self.led_chip = gpiod.Chip('gpiochip3')
                print(f"✅ 成功打开GPIO芯片: {self.led_chip.name}")
                print(f"🔍 GPIO芯片信息: {self.led_chip.num_lines}条线路")
            except Exception as e:
                print(f"❌ 打开GPIO芯片失败: {e}")
                self.led_chip = None
                return
            
            # 获取GPIO线路
            try:
                self.led_red = self.led_chip.get_line(4)
                self.led_green = self.led_chip.get_line(0)
                self.led_blue = self.led_chip.get_line(3)
                print(f"✅ 成功获取GPIO线路: red(4), green(0), blue(3)")
            except Exception as e:
                print(f"❌ 获取GPIO线路失败: {e}")
                self.led_red = None
                self.led_green = None
                self.led_blue = None
                return
            
            # 请求GPIO线路
            try:
                self.led_red.request(consumer="LED", type=gpiod.LINE_REQ_DIR_OUT)
                self.led_green.request(consumer="LED", type=gpiod.LINE_REQ_DIR_OUT)
                self.led_blue.request(consumer="LED", type=gpiod.LINE_REQ_DIR_OUT)
                print("✅ 成功请求GPIO线路控制权")
            except Exception as e:
                print(f"❌ 请求GPIO线路控制权失败: {e}")
                self.led_red = None
                self.led_green = None
                self.led_blue = None
                return
            
            # 初始化时关闭LED
            try:
                self.led_red.set_value(0)
                self.led_green.set_value(0)
                self.led_blue.set_value(0)
                print("✅ 成功初始化LED为关闭状态")
            except Exception as e:
                print(f"⚠️ 初始化LED状态失败: {e}")
            
            print("✅ LED控制初始化成功")
            
            # 验证LED控制是否正常工作
            try:
                r_val = self.led_red.get_value()
                g_val = self.led_green.get_value()
                b_val = self.led_blue.get_value()
                print(f"🔍 当前LED状态: RGB({r_val},{g_val},{b_val})")
            except Exception as e:
                print(f"⚠️ 无法获取LED状态: {e}")
            
        except Exception as e:
            print(f"⚠️ LED控制初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self.led_chip = None
            self.led_red = None
            self.led_green = None
            self.led_blue = None
    
    def set_led_color(self, status):
        """设置LED颜色"""
        print(f"🔍 尝试设置LED颜色为: {status}")
        
        if not self.led_red or not self.led_green or not self.led_blue:
            print("⚠️ LED未正确初始化，跳过设置")
            print(f"  LED状态: red={self.led_red}, green={self.led_green}, blue={self.led_blue}")
            return
        
        try:
            print(f"🔍 LED颜色映射: {self.led_colors}")
            if status in self.led_colors:
                r, g, b = self.led_colors[status]
                print(f"🔍 获取到颜色映射: {status} -> RGB({r},{g},{b})")
                
                # 直接打印当前LED值
                try:
                    print(f"🔍 当前LED值: red={self.led_red.get_value()}, green={self.led_green.get_value()}, blue={self.led_blue.get_value()}")
                except:
                    print("🔍 无法获取当前LED值")
                
                # 设置新值
                self.led_red.set_value(r)
                self.led_green.set_value(g)
                self.led_blue.set_value(b)
                print(f"💡 LED设置为{status}色: RGB({r},{g},{b})")
                
                # 验证设置是否成功
                try:
                    new_r = self.led_red.get_value()
                    new_g = self.led_green.get_value()
                    new_b = self.led_blue.get_value()
                    print(f"🔍 设置后LED值: red={new_r}, green={new_g}, blue={new_b}")
                    if new_r != r or new_g != g or new_b != b:
                        print(f"⚠️ LED设置不匹配: 期望RGB({r},{g},{b}), 实际RGB({new_r},{new_g},{new_b})")
                except:
                    print("🔍 无法获取设置后的LED值")
            else:
                print(f"⚠️ 未知状态: {status}")
                print(f"🔍 可用状态: {list(self.led_colors.keys())}")
                
        except Exception as e:
            print(f"⚠️ LED设置失败: {e}")
            import traceback
            traceback.print_exc()
    
    def cleanup_communication_files(self):
        """清理进程间通信文件"""
        for file_path in [self.status_file, self.command_file, self.voice_trigger_file]:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
    
    def signal_handler(self, signum, frame):
        """处理系统信号"""
        print(f"\n接收到信号 {signum}，正在退出...")
        self.running = False
        sys.exit(0)
    
    def check_voice_trigger(self):
        """检查语音触发 - 移除此功能，由语音监控器处理"""
        # 此功能已移至语音监控器
        return False
    
    def save_current_status(self):
        """保存当前交通状态到文件"""
        try:
            status_data = {
                'timestamp': time.time(),
                'last_prediction': getattr(self, 'last_prediction', '等待中...'),
                'last_confidence': getattr(self, 'last_confidence', 0.0),
                'last_announced_status': self.last_announced_status,
                'status_change_count': self.status_change_count,
                'current_tracking_status': getattr(self, 'current_tracking_status', None),
                'prediction_history': list(self.prediction_history)
            }
            
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(status_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 状态已保存到 {self.status_file}")
            
        except Exception as e:
            pass
            #print(f"保存状态失败: {e}")
    
    def start_voice_processing(self):
        """启动语音处理程序 - 移除此功能，由语音监控器处理"""
        # 此功能已移至语音监控器
        pass
    
    def chinese_to_pinyin(self, text):
        """将中文转换为拼音"""
        try:
            pinyin_list = pinyin(text)
            pinyin_text = ' '.join([item[0] for item in pinyin_list])
            return pinyin_text
        except Exception as e:
            print(f"拼音转换失败: {e}")
            return "speech error"
    
    async def speak_text(self, text):
        """使用espeak播放中文文本"""
        try:
            communicate = Communicate(text=text, voice="zh-CN-XiaoxiaoNeural")
            await communicate.save("tts_output.mp3")
    
            playsound("tts_output.mp3")
            
            
        except Exception as e:
            print(f"语音播放失败: {e}")
    
    def load_model(self):
        """加载RKNN模型"""
        try:
            print("加载RKNN模型...")
            self.rknn = RKNN(verbose=False)
            
            ret = self.rknn.load_rknn(self.model_path)
            if ret != 0:
                print(f"加载RKNN模型失败，错误码: {ret}")
                return False
            
            print("初始化运行时环境...")
            ret = self.rknn.init_runtime(target='rk3588')
            if ret != 0:
                print(f"初始化运行时环境失败，错误码: {ret}")
                return False
            
            print("✅ RKNN模型加载成功!")
            return True
            
        except Exception as e:
            print(f"加载模型时发生错误: {e}")
            return False
    
    def init_camera(self):
        """初始化摄像头"""
        try:
            print(f"初始化摄像头端口 {self.camera_port}...")
            self.cap = cv2.VideoCapture(self.camera_port)
            
            if not self.cap.isOpened():
                print(f"无法打开摄像头端口 {self.camera_port}")
                return False
            
            # 设置摄像头参数 - 720p分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            ret, frame = self.cap.read()
            if not ret:
                print("摄像头无法读取帧")
                return False
            
            print(f"✅ 摄像头初始化成功! 分辨率: {frame.shape[1]}x{frame.shape[0]}")
            return True
            
        except Exception as e:
            print(f"初始化摄像头时发生错误: {e}")
            return False
    
    def preprocess_frame(self, frame):
        """预处理单帧图像"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        for c in range(3):
            normalized[:, :, c] = (normalized[:, :, c] - mean[c]) / std[c]
        
        return normalized
    
    def extract_frames_from_buffer(self, num_frames=5):
        """从帧缓冲区提取帧序列"""
        if len(self.frame_buffer) < num_frames:
            frames = list(self.frame_buffer)
            while len(frames) < num_frames:
                if len(frames) > 0:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.float32))
            return [frames]
        
        sequences = []
        buffer_list = list(self.frame_buffer)
        total_frames = len(buffer_list)
        
        # 最新的5帧
        if total_frames >= num_frames:
            latest_sequence = buffer_list[-num_frames:]
            sequences.append(latest_sequence)
        
        # 中间的5帧
        if total_frames >= num_frames * 2:
            middle_start = total_frames // 2 - num_frames // 2
            middle_sequence = buffer_list[middle_start:middle_start + num_frames]
            sequences.append(middle_sequence)
        
        # 较早的5帧
        if total_frames >= num_frames * 3:
            early_sequence = buffer_list[:num_frames]
            sequences.append(early_sequence)
        
        return sequences if sequences else [[np.zeros((224, 224, 3), dtype=np.float32)] * num_frames]
    
    def predict_sequences(self, sequences):
        """对帧序列进行预测"""
        all_predictions = []
        all_confidences = []
        
        for i, sequence in enumerate(sequences):
            try:
                input_tensor = np.zeros((1, 5, 3, 224, 224), dtype=np.float32)
                
                for j, frame in enumerate(sequence):
                    frame_chw = np.transpose(frame, (2, 0, 1))
                    input_tensor[0, j] = frame_chw
                
                outputs = self.rknn.inference(inputs=[input_tensor])
                
                if outputs and len(outputs) > 0:
                    logits = outputs[0][0]
                    exp_logits = np.exp(logits - np.max(logits))
                    probabilities = exp_logits / np.sum(exp_logits)
                    
                    prediction = np.argmax(probabilities)
                    confidence = probabilities[prediction]
                    
                    all_predictions.append(prediction)
                    all_confidences.append(confidence)
                    
                    print(f"序列{i+1}: 预测={self.status_map[prediction]}, 置信度={confidence:.4f}")
                else:
                    print(f"序列{i+1}: 推理失败")
                    
            except Exception as e:
                print(f"序列{i+1}预测时发生错误: {e}")
                continue
        
        if not all_predictions:
            return 0, 0.0, [], []
        
        # 加权投票
        weighted_votes = {}
        for pred, conf in zip(all_predictions, all_confidences):
            if pred not in weighted_votes:
                weighted_votes[pred] = 0
            weighted_votes[pred] += conf
        
        final_prediction = max(weighted_votes.items(), key=lambda x: x[1])[0]
        avg_confidence = sum(all_confidences) / len(all_confidences)
        
        return final_prediction, avg_confidence, all_predictions, all_confidences
    
    def check_status_change_and_announce(self, current_status, confidence):
        """检查状态变化并进行语音播报"""
        if self.last_announced_status is None:
            if hasattr(self, 'current_tracking_status') and self.current_tracking_status == current_status:
                self.status_change_count += 1
            else:
                self.current_tracking_status = current_status
                self.status_change_count = 1
            print(f"初始状态追踪: {current_status} (计数: {self.status_change_count}/{self.min_status_duration})")
        else:
            if self.last_announced_status != current_status:
                if hasattr(self, 'current_tracking_status') and self.current_tracking_status == current_status:
                    self.status_change_count += 1
                else:
                    self.current_tracking_status = current_status
                    self.status_change_count = 1
                print(f"状态变化追踪: {self.last_announced_status} -> {current_status} (计数: {self.status_change_count}/{self.min_status_duration})")
            else:
                self.status_change_count = 0
                self.current_tracking_status = current_status
        
        # 更新LED颜色 - 实时显示当前预测状态
        self.set_led_color(current_status)
        
        should_announce = False
        
        if self.status_change_count >= self.min_status_duration:
            if self.last_announced_status is None:
                should_announce = True
                print(f"✅ 首次播报条件满足: {current_status}")
            elif self.last_announced_status != current_status:
                should_announce = True
                print(f"✅ 状态变化播报条件满足: {self.last_announced_status} -> {current_status}")
        
        if should_announce:
            prompt = self.voice_prompts.get(current_status, "检测到路况变化")
            print(f"🔊 语音播报: {prompt}")
            
            asyncio.run(self.speak_text(prompt))
            
            self.last_announced_status = current_status
            self.status_change_count = 0
    
    def capture_frames(self):
        """摄像头帧捕获线程"""
        while self.running:
            try:
                if self.prediction_paused:
                    time.sleep(0.1)
                    continue
                
                ret, frame = self.cap.read()
                if ret:
                    processed_frame = self.preprocess_frame(frame)
                    self.frame_buffer.append(processed_frame)
                    
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.frame_queue.put(frame)
                else:
                    print("读取摄像头帧失败")
                    time.sleep(0.1)
                    
                time.sleep(0.03)
                
            except Exception as e:
                if self.running:
                    print(f"帧捕获错误: {e}")
                time.sleep(0.1)
    
    def display_camera_feed(self):
        """显示摄像头画面"""
        last_display_time = 0
        display_interval = 1.0 / 30.0
        
        while self.running:
            try:
                current_time = time.time()
                
                if current_time - last_display_time < display_interval:
                    time.sleep(0.01)
                    continue
                
                display_frame = None
                try:
                    while not self.frame_queue.empty():
                        try:
                            display_frame = self.frame_queue.get_nowait()
                        except queue.Empty:
                            break
                except Exception as e:
                    if self.running:
                        print(f"获取显示帧错误: {e}")
                
                if display_frame is not None:
                    # 显示预测结果
                    if hasattr(self, 'last_prediction'):
                        result_text = f"{self.last_prediction} ({self.last_confidence:.2f})"
                        cv2.putText(display_frame, result_text, (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # 显示状态计数
                    if hasattr(self, 'current_tracking_status'):
                        count_text = f"Status: {self.current_tracking_status} ({self.status_change_count}/{self.min_status_duration})"
                        cv2.putText(display_frame, count_text, (10, 70), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # 显示程序状态
                    status_text = "PREDICTION MODE - Press 'q' to quit"
                    cv2.putText(display_frame, status_text, (10, 100), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # 显示帧率
                    fps_text = f"FPS: {1.0/(current_time - last_display_time):.1f}" if last_display_time > 0 else "FPS: --"
                    cv2.putText(display_frame, fps_text, (10, 130), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.imshow('Traffic Prediction', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("用户请求退出...")
                        self.running = False
                        break
                    elif key == ord('s'):
                        timestamp = time.strftime('%Y%m%d_%H%M%S')
                        filename = f'traffic_frame_{timestamp}.jpg'
                        cv2.imwrite(filename, display_frame)
                        print(f"保存帧: {filename}")
                    elif key == ord('p'):
                        self.prediction_paused = not self.prediction_paused
                        print(f"预测{'暂停' if self.prediction_paused else '恢复'}")
                
                last_display_time = current_time
                
            except Exception as e:
                if self.running:
                    print(f"显示错误: {e}")
                time.sleep(0.1)
    
    def run_prediction(self):
        """主要的预测运行循环 - 移除语音监控功能"""
        print("开始预测循环...")
        prediction_count = 0
        
        while self.running:
            try:
                # 移除语音触发检查，专注于预测
                
                if self.prediction_paused:
                    print("⏸️ 预测已暂停...")
                    time.sleep(0.5)
                    continue
                
                if len(self.frame_buffer) < 5:
                    print("等待更多帧数据...")
                    time.sleep(1)
                    continue
                
                print(f"\n{'='*60}")
                print(f"预测 #{prediction_count + 1} - {time.strftime('%H:%M:%S')}")
                print(f"{'='*60}")
                
                # 提取帧序列
                sequences = self.extract_frames_from_buffer()
                print(f"提取了 {len(sequences)} 个序列进行预测")
                
                # 进行预测
                prediction, confidence, all_preds, all_confs = self.predict_sequences(sequences)
                
                # 使用原始预测结果
                raw_status = self.status_map[prediction]
                
                # 显示预测结果
                print(f"\n📊 预测结果:")
                print(f"  当前预测: {raw_status} (置信度: {confidence:.4f})")
                
                if len(all_preds) > 1:
                    print(f"  序列预测分布: {Counter(all_preds)}")
                    print(f"  平均置信度: {np.mean(all_confs):.4f}")
                
                # 显示状态播报信息
                if self.last_announced_status is None:
                    print(f"  初始状态检测: {raw_status} (计数: {self.status_change_count}/{self.min_status_duration})")
                else:
                    print(f"  上次播报状态: {self.last_announced_status}")
                    if hasattr(self, 'current_tracking_status'):
                        print(f"  当前追踪状态: {self.current_tracking_status} (计数: {self.status_change_count}/{self.min_status_duration})")
                
                # 检查状态变化并播报
                print(f"🔍 将状态 '{raw_status}' 传递给 check_status_change_and_announce 函数")
                print(f"🔍 状态映射: {self.status_map}")
                print(f"🔍 LED颜色映射: {self.led_colors}")
                self.check_status_change_and_announce(raw_status, confidence)
                
                # 保存预测结果
                self.last_prediction = raw_status
                self.last_confidence = confidence
                self.prediction_history.append((prediction, confidence))
                
                # 自动保存状态变化
                self.save_current_status()
                
                prediction_count += 1
                
                # 每2秒进行一次预测
                time.sleep(2)
                
            except KeyboardInterrupt:
                print("\n接收到中断信号，正在退出...")
                break
            except Exception as e:
                print(f"预测循环中发生错误: {e}")
                time.sleep(1)
        
        print("预测循环结束")
    
    def test_led(self):
        """测试LED控制"""
        print("🔧 测试LED控制...")
        
        if not self.led_red or not self.led_green or not self.led_blue:
            print("⚠️ LED未正确初始化，无法测试")
            return False
        
        try:
            # 测试所有交通状态颜色
            for status, (r, g, b) in self.led_colors.items():
                print(f"测试LED颜色: {status} -> RGB({r},{g},{b})")
                self.led_red.set_value(r)
                self.led_green.set_value(g)
                self.led_blue.set_value(b)
                time.sleep(0.5)  # 短暂延迟以便观察
            
            # 恢复为关闭状态
            self.led_red.set_value(0)
            self.led_green.set_value(0)
            self.led_blue.set_value(0)
            
            print("✅ LED测试完成")
            return True
            
        except Exception as e:
            print(f"❌ LED测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def start(self):
        """启动交通预测系统"""
        print("🚀 启动交通预测系统（独立进程模式）...")
        
        # 设置信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # 初始化资源
        if not self.load_model():
            print("❌ 模型加载失败，无法启动")
            return False
        
        if not self.init_camera():
            print("❌ 摄像头初始化失败，无法启动")
            return False
        
        # 测试LED控制
        self.test_led()
        
        # 初始化预测结果变量
        self.last_prediction = "等待中..."
        self.last_confidence = 0.0
        
        # 设置运行标志
        self.running = True
        
        # 启动线程
        capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        capture_thread.start()
        print("✅ 帧捕获线程已启动")
        
        display_thread = threading.Thread(target=self.display_camera_feed, daemon=True)
        display_thread.start()
        print("✅ 显示线程已启动")
        
        # 等待缓冲区填充
        print("⏳ 等待帧缓冲区填充...")
        time.sleep(3)
        
        # 启动预测循环
        try:
            self.run_prediction()
        except Exception as e:
            print(f"预测系统运行错误: {e}")
        finally:
            self.stop()
        
        return True
    
    def stop(self):
        """停止预测系统"""
        print("🔄 正在停止交通预测系统...")
        
        self.running = False
        
        # 关闭LED
        if self.led_chip:
            try:
                self.set_led_color('关闭')
                print("💡 LED已关闭")
            except Exception as e:
                print(f"⚠️ LED关闭失败: {e}")
        
        if self.rknn:
            self.rknn.release()
            print("✅ RKNN模型已释放")
        
        if self.cap:
            self.cap.release()
            print("✅ 摄像头已关闭")
        
        cv2.destroyAllWindows()
        print("✅ 系统已完全停止")
    
    def cleanup(self):
        """清理资源和通信文件"""
        print("🔄 清理资源...")
        
        # 停止预测系统
        self.stop()
        
        # 清理通信文件
        self.cleanup_communication_files()
        
        print("✅ 资源清理完成")

def main():
    """主函数 - 修改为纯预测模式"""
    print("=" * 60)
    print("ELF2交通预测程序 - 独立进程模式")
    print("=" * 60)
    
    # 模型路径配置
    model_path = '/userdata/deepseek/python/traffic_model_rk3588.rknn'
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件不存在 - {model_path}")
        print("请确保模型文件路径正确")
        return
    
    # 创建交通预测器实例
    predictor = TrafficPredictionMain(
        model_path=model_path,
        camera_port=21
    )
    
    print("\n🎯 程序功能:")
    print("- 实时交通状况预测")
    print("- 720p分辨率原始预测结果")
    print("- 状态变化自动保存")
    print("- 支持四种交通状况：畅通、缓慢、拥堵、封闭")
    print("- 独立进程运行，专注预测功能")
    print("- 摄像头端口: 21")
    print()
    print("🔧 工作流程:")
    print("1. 初始化RKNN模型和摄像头")
    print("2. 连续捕获和预测交通状况")
    print("3. 实时显示预测结果")
    print("4. 自动保存状态变化")
    print("5. 等待外部程序管理")
    print()
    print("📡 状态输出:")
    print("- 交通状态文件: /tmp/traffic_status.json")
    print("- 实时预测显示")
    print("- 状态变化日志")
    print()
    print("🎮 控制方式:")
    print("- 按 'q' 键退出")
    print("- 按 's' 键手动保存状态")
    print("- Ctrl+C 优雅退出")
    print()
    print("🧠 预测逻辑:")
    print("- 使用原始预测结果，无平滑处理")
    print("- 实时响应交通状况变化")
    print("- 三次连续相同预测触发状态变化")
    print("=" * 60)
    
    try:
        # 设置信号处理器
        signal.signal(signal.SIGINT, predictor.signal_handler)
        signal.signal(signal.SIGTERM, predictor.signal_handler)
        
        # 启动预测系统
        predictor.start()
        
    except KeyboardInterrupt:
        print("\n接收到中断信号")
    except Exception as e:
        print(f"程序运行错误: {e}")
    finally:
        predictor.cleanup()
        print("程序结束")

if __name__ == "__main__":
    main() 
