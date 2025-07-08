import os
import time
import json
import signal
import sys
import subprocess
import shlex
import serial
import re
from pypinyin import pinyin, Style
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

class VoiceProcessingMain:
    def __init__(self, audio_port='/dev/ttyS9', map_file='/userdata/deepseek/python/map.txt'):
        """
        语音处理主程序（独立进程）
        
        Args:
            audio_port: 语音模块串口路径
            map_file: 地图数据文件路径
        """
        self.audio_port = audio_port
        self.map_file = map_file
        self.audio_serial = None
        self.deepseek_process = None
        self.running = False
        
        # LED控制初始化
        self.led_chip = None
        self.led_red = None
        self.led_green = None
        self.led_blue = None
        self.init_led()
        
        # 设置LED为紫色 - 语音模式
        self.set_led_color(1, 0, 1)
        
        # RAG相关数据存储
        self.map_data = []
        self.route_index = {}
        
        # 进程间通信文件
        self.status_file = '/tmp/traffic_status.json'
        self.command_file = '/tmp/traffic_command.txt'
        self.voice_input_file = '/tmp/voice_input.txt'
        self.voice_trigger_file = '/tmp/voice_trigger.txt'
        
        # 当前交通状态（从预测程序加载）
        self.current_traffic_status = None
        self.prediction_history = []
        
        # 加载地图数据
        self.load_map_data()
        
        # 加载交通状态
        self.load_traffic_status()
        
        print(f"语音处理主程序初始化 - 音频端口: {audio_port}")
    
    def init_led(self):
        """初始化LED控制"""
        if not LED_AVAILABLE:
            print("⚠️ LED控制不可用，跳过LED初始化")
            return
        
        try:
            print("🔧 初始化LED控制...")
            self.led_chip = gpiod.Chip('gpiochip3')
            self.led_red = self.led_chip.get_line(4)
            self.led_green = self.led_chip.get_line(0)
            self.led_blue = self.led_chip.get_line(3)
            
            # 不在初始化时请求LED控制权，而是在需要使用时请求
            print("✅ LED控制初始化成功")
            
        except Exception as e:
            print(f"⚠️ LED控制初始化失败: {e}")
            self.led_chip = None
            self.led_red = None
            self.led_green = None
            self.led_blue = None
    
    def set_led_color(self, r, g, b):
        """设置LED颜色，设置后释放控制权"""
        if not self.led_red or not self.led_green or not self.led_blue:
            print("⚠️ LED未正确初始化，跳过设置")
            return
        
        try:
            # 请求LED控制权
            self.led_red.request(consumer="LED", type=gpiod.LINE_REQ_DIR_OUT)
            self.led_green.request(consumer="LED", type=gpiod.LINE_REQ_DIR_OUT)
            self.led_blue.request(consumer="LED", type=gpiod.LINE_REQ_DIR_OUT)
            
            # 设置LED颜色
            self.led_red.set_value(r)
            self.led_green.set_value(g)
            self.led_blue.set_value(b)
            print(f"💡 LED设置为: RGB({r},{g},{b})")
            
            # 释放LED控制权
            self.led_red.release()
            self.led_green.release()
            self.led_blue.release()
            print("💡 LED控制权已释放")
                
        except Exception as e:
            print(f"⚠️ LED设置失败: {e}")
            # 确保发生异常时也释放控制权
            try:
                self.led_red.release()
                self.led_green.release()
                self.led_blue.release()
            except:
                pass
    
    def signal_handler(self, signum, frame):
        """处理系统信号"""
        print(f"\n接收到信号 {signum}，正在退出...")
        self.running = False
        self.cleanup_and_restart_prediction()
        sys.exit(0)
    
    def load_map_data(self):
        """加载地图数据并构建索引"""
        try:
            print(f"加载地图数据: {self.map_file}")
            
            if not os.path.exists(self.map_file):
                print(f"警告: 地图文件不存在 - {self.map_file}")
                return
            
            with open(self.map_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if not line or line.endswith('：'):
                    continue
                
                if '从' in line and '到' in line and '途经' in line:
                    match = re.search(r'从(.+?)到(.+?)，', line)
                    if match:
                        start_point = match.group(1).strip()
                        end_point = match.group(2).strip()
                        
                        route_part = line.split('途经')[1] if '途经' in line else line.split('途径')[1]
                        waypoints = [point.strip() for point in route_part.split('、')]
                        
                        route_info = {
                            'start': start_point,
                            'end': end_point,
                            'waypoints': waypoints,
                            'description': line
                        }
                        
                        self.map_data.append(route_info)
                        
                        for location in [start_point, end_point] + waypoints:
                            if location not in self.route_index:
                                self.route_index[location] = []
                            self.route_index[location].append(route_info)
            
            print(f"✅ 成功加载 {len(self.map_data)} 条路线信息")
            print(f"✅ 索引覆盖 {len(self.route_index)} 个地点")
            
        except Exception as e:
            print(f"加载地图数据失败: {e}")
    
    def load_traffic_status(self):
        """加载当前交通状态"""
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    status_data = json.load(f)
                
                self.current_traffic_status = status_data.get('last_prediction', '未知')
                self.prediction_history = status_data.get('prediction_history', [])
                
                print(f"✅ 加载交通状态: {self.current_traffic_status}")
                print(f"✅ 加载预测历史: {len(self.prediction_history)} 条记录")
            else:
                print("⚠️ 未找到交通状态文件，使用默认状态")
                self.current_traffic_status = "未知"
                
        except Exception as e:
            print(f"加载交通状态失败: {e}")
            self.current_traffic_status = "未知"
    
    def search_routes(self, query):
        """基于查询搜索相关路线信息"""
        relevant_routes = []
        query_lower = query.lower()
        
        for location, routes in self.route_index.items():
            if location in query:
                relevant_routes.extend(routes)
        
        # 去重
        unique_routes = []
        seen = set()
        for route in relevant_routes:
            route_key = f"{route['start']}->{route['end']}"
            if route_key not in seen:
                unique_routes.append(route)
                seen.add(route_key)
        
        return unique_routes
    
    def build_rag_context(self, query):
        """构建RAG上下文"""
        relevant_routes = self.search_routes(query)
        
        if not relevant_routes:
            return "当前没有找到相关的路线信息。"
        
        context = "相关路线信息：\n"
        for i, route in enumerate(relevant_routes[:5], 1):
            context += f"{i}. {route['description']}\n"
        
        if self.current_traffic_status:
            context += f"\n当前路况: {self.current_traffic_status}\n"
        
        return context
    
    def format_rag_prompt(self, user_question, context):
        """格式化RAG提示词"""
        prompt = f"""基于以下地图和路况信息:{context}，请回答用户的问题：,用户问题: {user_question} ,应该怎么走,请提供准确、有用的回答，如果涉及路线规划，请参考上述信息。回答请简洁明了,不要分条，一句话作答，不要用*"""
        prompt = prompt.replace('\n', '')
        
        return prompt
    
    def init_audio_serial(self):
        """初始化语音模块串口"""
        try:
            print(f"初始化语音模块串口 {self.audio_port}...")
            self.audio_serial = serial.Serial(
                self.audio_port, 
                9600,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1
            )
            print("✅ 语音模块串口初始化成功!")
            return True
        except Exception as e:
            print(f"❌ 初始化语音模块串口失败: {e}")
            return False
    
    def check_memory_status(self):
        """检查当前内存状态"""
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            
            lines = meminfo.strip().split('\n')
            mem_total = 0
            mem_free = 0
            mem_available = 0
            mem_buffers = 0
            mem_cached = 0
            
            for line in lines:
                if line.startswith('MemTotal:'):
                    mem_total = int(line.split()[1])
                elif line.startswith('MemFree:'):
                    mem_free = int(line.split()[1])
                elif line.startswith('MemAvailable:'):
                    mem_available = int(line.split()[1])
                elif line.startswith('Buffers:'):
                    mem_buffers = int(line.split()[1])
                elif line.startswith('Cached:'):
                    mem_cached = int(line.split()[1])
            
            # 转换为MB
            mem_total_mb = mem_total // 1024
            mem_free_mb = mem_free // 1024
            mem_available_mb = mem_available // 1024
            mem_buffers_mb = mem_buffers // 1024
            mem_cached_mb = mem_cached // 1024
            mem_used_mb = mem_total_mb - mem_free_mb
            
            print(f"💾 内存状态:")
            print(f"  总内存: {mem_total_mb} MB")
            print(f"  已用内存: {mem_used_mb} MB")
            print(f"  可用内存: {mem_available_mb} MB")
            print(f"  实际使用: {mem_used_mb - mem_buffers_mb - mem_cached_mb} MB")
            
            # 检查可用内存是否足够
            min_required_mb = 1024
            if mem_available_mb < min_required_mb:
                print(f"⚠️ 警告: 可用内存 {mem_available_mb} MB 可能不足以启动DeepSeek")
                return False
            else:
                print(f"✅ 内存充足，可以启动DeepSeek")
                return True
                
        except Exception as e:
            print(f"⚠️ 无法检查内存状态: {e}")
            return True
    
    def init_deepseek(self):
        """初始化DeepSeek"""
        print("🚀 启动DeepSeek...")
        
        # 检查内存状态
        if not self.check_memory_status():
            print("❌ 内存不足，无法启动DeepSeek")
            return False
        
        try:
            cmd = "/userdata/deepseek/python/llm_demo DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm 4096 4096"
            print(f"执行命令: {cmd}")
            
            self.deepseek_process = subprocess.Popen(
                shlex.split(cmd), 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE
            )
            print("⏳ 等待DeepSeek初始化...")
            time.sleep(20)
            print(self.deepseek_process.stdout.read1().decode('utf-8'))
            
            # 检查进程是否还活着
            if self.deepseek_process.poll() is not None:
                stderr_output = self.deepseek_process.stderr.read().decode('utf-8')
                stdout_output = self.deepseek_process.stdout.read().decode('utf-8')
                print(f"❌ DeepSeek进程意外退出")
                print(f"  错误输出: {stderr_output}")
                print(f"  标准输出: {stdout_output}")
                return False
            
            return True
                
        except Exception as e:
            print(f"❌ DeepSeek启动失败: {e}")
            if self.deepseek_process:
                try:
                    self.deepseek_process.terminate()
                except:
                    pass
                self.deepseek_process = None
            return False
    
    def test_deepseek(self):
        """测试DeepSeek是否正常工作"""
        try:
            print("测试DeepSeek连接...")
            test_question = "你好"
            
            question_with_newline = test_question + '\n'
            self.deepseek_process.stdin.write(question_with_newline.encode('utf-8'))
            self.deepseek_process.stdin.flush()
            
            feedback = ""
            start_time = time.time()
            timeout = 10
            
            while time.time() - start_time < timeout:
                try:
                    tmp = self.deepseek_process.stdout.read1().decode('utf-8')
                    if tmp:
                        feedback += tmp
                        if len(feedback) > 50 or "user:" in tmp:
                            break
                    else:
                        time.sleep(0.1)
                except Exception as e:
                    print(f"读取测试输出错误: {e}")
                    break
            
            print(f"测试回答: {feedback[:100]}...")
            return len(feedback) > 0
            
        except Exception as e:
            print(f"DeepSeek测试失败: {e}")
            return False
            
    def clean_advice(self, text):
        """clean think part"""
        pattern = r'<think>.*?</think>'
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
        
        cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)
        
        return cleaned_text.strip()
    
    def query_deepseek(self, question):
        """向DeepSeek查询并获取回答"""
        try:
            if not self.deepseek_process:
                return "DeepSeek未初始化"
            
            print(f"用户问题: {question}")
            
            # 构建RAG上下文
            context = self.build_rag_context(question)
            print(f"检索到的上下文: {context}")
            
            # 格式化RAG提示词
            rag_prompt = self.format_rag_prompt(question, context)
            print(f"RAG提示词长度: {len(rag_prompt)} 字符")
            
            # 发送问题到DeepSeek
            question_with_newline = rag_prompt + '\n'
            #print("question: ", question_with_newline)
            self.deepseek_process.stdin.write(question_with_newline.encode())
            self.deepseek_process.stdin.flush()
            
            # 读取回答
            feedback = ""
            start_time = time.time()
            timeout = 30
            
            while time.time() - start_time < timeout:
                tmp = self.deepseek_process.stdout.read1().decode()
                if "user:" in tmp:
                    feedback += tmp
                    break
                else:
                    feedback += tmp
                    time.sleep(0.1)
            
            # 提取回答内容
            #print("feedback: ", feedback)
            if len(feedback) > 33:
                feedback = self.clean_advice(feedback)
                answer = feedback[9:-7].strip()
                print(f"DeepSeek原始回答: {answer}")
                
                # 后处理回答
                if len(answer) > 150:
                    answer = answer[:150] + "..."
                
                return answer
            else:
                return "获取回答失败"
                
        except Exception as e:
            print(f"DeepSeek查询失败: {e}")
            return "查询失败"
    
    def chinese_to_pinyin(self, text):
        """将中文转换为拼音"""
        try:
            pinyin_list = pinyin(text, style=Style.NORMAL, heteronym=False)
            pinyin_text = ' '.join([item[0] for item in pinyin_list])
            return pinyin_text
        except Exception as e:
            print(f"拼音转换失败: {e}")
            return "speech error"
    
    async def speak_text(self, text):
        """使用espeak播放中文文本"""
        try:
            communicate = Communicate(text=text, voice="zh-CN-XiaoxiaoNeural")
            await communicate.save("tts_output_deepseek.mp3")
    
            playsound("tts_output_deepseek.mp3")
            
            
        except Exception as e:
            print(f"语音播放失败: {e}")
    
    def monitor_voice_input(self):
        """监控语音输入 - 修改为从文件读取"""
        print("🎤 开始处理语音输入...")
        
        try:
            # 读取语音输入文件
            voice_input_file = '/tmp/voice_input.txt'
            if not os.path.exists(voice_input_file):
                print("❌ 语音输入文件不存在")
                return
            
            with open(voice_input_file, 'r', encoding='utf-8') as f:
                voice_text = f.read().strip()
            
            if not voice_text:
                print("❌ 语音输入为空")
                return
            
            print(f"🎤 读取到语音输入: {voice_text}")
            
            # 初始化DeepSeek（如果还未初始化）
            if not self.deepseek_process:
                print("🚀 初始化DeepSeek...")
                if not self.init_deepseek():
                    print("❌ DeepSeek初始化失败")
                    return
            
            # 处理语音输入
            success = self.process_voice_input(voice_text)
            
            if success:
                print("✅ 语音处理完成")
            else:
                print("❌ 语音处理失败")
                error_msg = "抱歉，语音处理出现错误"
                asyncio.run(self.speak_text(error_msg))
            
        except Exception as e:
            print(f"❌ 处理语音输入失败: {e}")
            error_msg = "抱歉，系统出现错误"
            asyncio.run(self.speak_text(error_msg))
        finally:
            # 清理临时文件
            try:
                if os.path.exists('/tmp/voice_input.txt'):
                    os.remove('/tmp/voice_input.txt')
                    print("✅ 清理语音输入文件")
            except Exception as e:
                print(f"⚠️ 清理语音输入文件失败: {e}")
    
    def process_voice_input(self, voice_text):
        """处理语音输入"""
        try:
            print(f"🎤 处理语音输入: {voice_text}")
            
            # 加载最新的交通状态
            self.load_traffic_status()
            
            # 获取AI回答
            answer = self.query_deepseek(voice_text)
            print(f"🤖 AI回答: {answer}")
            
            if answer and answer != "获取回答失败" and answer != "查询失败":
                # 播放回答
                print("🔊 播放语音回答...")
                asyncio.run(self.speak_text(answer))
                return True
            else:
                print("❌ 未获得有效回答")
                fallback_msg = "抱歉，我无法理解您的问题，请重新提问"
                asyncio.run(self.speak_text(fallback_msg))
                return False
            
        except Exception as e:
            print(f"处理语音输入失败: {e}")
            return False
    
    def process_initial_voice_input(self):
        """处理初始语音输入（从文件读取）"""
        try:
            if os.path.exists(self.voice_input_file):
                with open(self.voice_input_file, 'r', encoding='utf-8') as f:
                    voice_text = f.read().strip()
                
                if voice_text:
                    print(f"🎤 处理初始语音输入: {voice_text}")
                    
                    # 删除输入文件
                    os.remove(self.voice_input_file)
                    
                    # 处理语音输入
                    success = self.process_voice_input(voice_text)
                    
                    if success:
                        print("✅ 初始语音处理完成")
                    else:
                        print("❌ 初始语音处理失败")
                        error_msg = "抱歉，语音处理出现错误"
                        asyncio.run(self.speak_text(error_msg))
                    
                    return True
                    
        except Exception as e:
            print(f"处理初始语音输入失败: {e}")
        
        return False
    
    def cleanup_and_restart_prediction(self):
        """清理资源并重启预测程序"""
        print("🔄 清理资源...")
        
        # 关闭LED - 使用新的LED控制方式
        if self.led_red and self.led_green and self.led_blue:
            try:
                # 直接调用set_led_color方法关闭LED
                self.set_led_color(0, 0, 0)
                print("💡 LED已关闭")
            except Exception as e:
                print(f"⚠️ LED关闭失败: {e}")
        
        # 关闭DeepSeek
        if self.deepseek_process:
            try:
                print("🔄 关闭DeepSeek...")
                self.deepseek_process.terminate()
                try:
                    self.deepseek_process.wait(timeout=5)
                    print("✅ DeepSeek已正常关闭")
                except subprocess.TimeoutExpired:
                    print("⚠️ DeepSeek超时，强制关闭")
                    self.deepseek_process.kill()
                    self.deepseek_process.wait()
                    print("✅ DeepSeek已强制关闭")
            except Exception as e:
                print(f"⚠️ DeepSeek关闭失败: {e}")
            finally:
                self.deepseek_process = None
        
        # 关闭语音串口
        if self.audio_serial:
            try:
                self.audio_serial.close()
                print("✅ 语音串口已关闭")
            except Exception as e:
                print(f"⚠️ 语音串口关闭失败: {e}")
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        print("✅ 资源清理完成")
        
        # 注意：不在这里重启预测程序，由语音监控器负责重启
    
    def start(self):
        """启动语音处理系统"""
        print("🚀 启动语音处理系统（独立进程模式）...")
        
        # 设置信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # 初始化语音模块
        audio_ok = self.init_audio_serial()
        if not audio_ok:
            print("❌ 语音模块初始化失败")
            return False
        
        # 初始化DeepSeek
        if not self.init_deepseek():
            print("❌ DeepSeek初始化失败")
            return False
        
        # 设置运行标志
        self.running = True
        
        # 首先处理初始语音输入
        if self.process_initial_voice_input():
            # 如果处理了初始输入，继续监控新的语音输入
            print("🎤 继续监控新的语音输入...")
            self.monitor_voice_input()
        else:
            # 如果没有初始输入，直接监控语音输入
            self.monitor_voice_input()
        
        # 清理并重启预测程序
        self.cleanup_and_restart_prediction()
        
        return True

def main():
    """主函数 - 修改为单次处理模式"""
    print("=" * 60)
    print("ELF2语音处理程序 - 智能交通助手")
    print("=" * 60)
    
    try:
        # 创建语音处理器实例
        print("🚀 创建语音处理器实例...")
        processor = VoiceProcessingMain(
            audio_port='/dev/ttyS9',
            map_file='/userdata/deepseek/python/map.txt'
        )
        print("✅ 语音处理器实例创建成功")
        
        print("\n🎯 程序功能:")
        print("- 读取语音输入文件进行处理")
        print("- 基于实时交通数据提供智能回复")
        print("- 支持路线查询和交通状况分析")
        print("- 处理完成后自动退出")
        print("- LED显示紫色表示语音处理模式")
        print()
        print("🔧 工作流程:")
        print("1. 读取语音输入文件 (/tmp/voice_input.txt)")
        print("2. 初始化DeepSeek AI")
        print("3. 加载交通状态和地图数据")
        print("4. 构建RAG上下文")
        print("5. 查询DeepSeek获取回复")
        print("6. 语音播报回复")
        print("7. 清理资源并退出")
        print()
        print("📡 通信方式:")
        print("- 语音输入: /tmp/voice_input.txt")
        print("- 交通状态: /tmp/traffic_status.json")
        print("- 地图数据: /userdata/deepseek/python/map.txt")
        print("- 语音输出: espeak TTS")
        print("- LED指示: 紫色(语音模式)")
        print()
        print("🧠 AI能力:")
        print("- 实时交通状况分析")
        print("- 路线规划建议")
        print("- 智能问答")
        print("- 上下文理解")
        print("=" * 60)
        
        # 设置信号处理器
        print("🔧 设置信号处理器...")
        signal.signal(signal.SIGINT, processor.signal_handler)
        signal.signal(signal.SIGTERM, processor.signal_handler)
        print("✅ 信号处理器设置完成")
        
        # 处理语音输入
        print("🎤 开始处理语音输入...")
        processor.monitor_voice_input()
        print("✅ 语音输入处理完成")
        
    except KeyboardInterrupt:
        print("\n接收到中断信号")
    except Exception as e:
        print(f"程序运行错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔄 开始清理资源...")
        try:
            if 'processor' in locals():
                processor.cleanup_and_restart_prediction()
        except Exception as e:
            print(f"清理资源时发生错误: {e}")
        print("程序结束")

if __name__ == "__main__":
    main() 
