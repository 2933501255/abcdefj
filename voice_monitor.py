import os
import time
import signal
import sys
import subprocess
import serial
import psutil

# 导入LED控制模块
try:
    import gpiod
    LED_AVAILABLE = True
    print("✅ LED控制模块导入成功")
except ImportError:
    LED_AVAILABLE = False
    print("⚠️ LED控制模块不可用")

class VoiceMonitor:
    def __init__(self, audio_port='/dev/ttyS9'):
        """
        语音监控程序 - 负责监控语音输入并触发程序切换
        
        Args:
            audio_port: 语音模块串口路径
        """
        self.audio_port = audio_port
        self.audio_serial = None
        self.running = False
        
        # LED控制初始化
        self.led_chip = None
        self.led_red = None
        self.led_green = None
        self.led_blue = None
        self.init_led()
        
        # 初始化时关闭LED
        self.set_led_color(0, 0, 0)
        
        # 进程间通信文件
        self.voice_trigger_file = '/tmp/voice_trigger.txt'
        
        # 当前运行的程序类型
        self.current_program = None  # 'prediction' 或 'voice'
        self.current_process = None
        
        print(f"语音监控程序初始化 - 音频端口: {audio_port}")
    
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
        self.cleanup()
        sys.exit(0)
    
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
            mem_available = 0
            
            for line in lines:
                if line.startswith('MemTotal:'):
                    mem_total = int(line.split()[1])
                elif line.startswith('MemAvailable:'):
                    mem_available = int(line.split()[1])
            
            # 转换为MB
            mem_total_mb = mem_total // 1024
            mem_available_mb = mem_available // 1024
            
            print(f"💾 内存状态: 总计 {mem_total_mb} MB, 可用 {mem_available_mb} MB")
            
            return mem_available_mb
                
        except Exception as e:
            print(f"⚠️ 无法检查内存状态: {e}")
            return 0
    
    def kill_existing_programs(self):
        """杀死现有的预测或语音处理程序"""
        try:
            print("🔄 检查并终止现有程序...")
            
            # 查找并终止相关进程
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and len(cmdline) > 1:
                        # 检查是否是我们的程序
                        if ('traffic_prediction_main.py' in ' '.join(cmdline) or 
                            'voice_processing_main.py' in ' '.join(cmdline)):
                            
                            print(f"🔄 终止进程: {proc.info['pid']} - {' '.join(cmdline)}")
                            proc.terminate()
                            
                            # 等待进程结束
                            try:
                                proc.wait(timeout=5)
                                print(f"✅ 进程 {proc.info['pid']} 已正常终止")
                            except psutil.TimeoutExpired:
                                print(f"⚠️ 进程 {proc.info['pid']} 超时，强制终止")
                                proc.kill()
                                proc.wait()
                                print(f"✅ 进程 {proc.info['pid']} 已强制终止")
                                
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                    
            # 等待一下让系统清理
            time.sleep(2)
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            print("✅ 程序清理完成")
            
        except Exception as e:
            print(f"⚠️ 程序清理失败: {e}")
    
    def start_prediction_program(self):
        """启动交通预测程序"""
        try:
            print("🚀 启动交通预测程序...")
            
            # 先终止现有程序
            self.kill_existing_programs()
            
            # 检查内存状态
            mem_available = self.check_memory_status()
            
            # 启动预测程序
            cmd = ['python3', 'traffic_prediction_main.py']
            self.current_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            self.current_program = 'prediction'
            print("✅ 交通预测程序已启动")
            
            # 等待程序稳定
            time.sleep(3)
            
            return True
            
        except Exception as e:
            print(f"❌ 启动交通预测程序失败: {e}")
            return False
    
    def start_voice_program(self, voice_text):
        """启动语音处理程序"""
        try:
            print(f"🚀 启动语音处理程序，处理: {voice_text}")
            
            # 先终止现有程序
            self.kill_existing_programs()
            
            # 设置LED为紫色 - 语音模式
            self.set_led_color(1, 0, 1)  # 紫色
            
            # 检查内存状态
            mem_available = self.check_memory_status()
            
            # 创建语音输入文件
            voice_input_file = '/tmp/voice_input.txt'
            with open(voice_input_file, 'w', encoding='utf-8') as f:
                f.write(voice_text)
            print(f"✅ 语音输入文件已创建: {voice_input_file}")
            
            # 启动语音处理程序
            cmd = ['python3', 'voice_processing_main.py']
            print(f"执行命令: {' '.join(cmd)}")
            
            self.current_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            self.current_program = 'voice'
            print(f"✅ 语音处理程序已启动，PID: {self.current_process.pid}")
            
            # 等待程序启动
            time.sleep(2)
            
            # 检查程序是否正常启动
            if self.current_process.poll() is None:
                print("✅ 语音处理程序运行正常")
                return True
            else:
                # 程序已经退出，读取错误信息
                stdout, stderr = self.current_process.communicate()
                print(f"❌ 语音处理程序启动失败")
                print(f"标准输出: {stdout}")
                print(f"错误输出: {stderr}")
                return False
            
        except Exception as e:
            print(f"❌ 启动语音处理程序失败: {e}")
            return False
    
    def monitor_current_program(self):
        """监控当前程序状态"""
        try:
            if self.current_process:
                # 检查进程是否还在运行
                poll_result = self.current_process.poll()
                if poll_result is not None:
                    print(f"🔄 {self.current_program} 程序已退出，返回码: {poll_result}")
                    
                    # 读取程序输出
                    try:
                        stdout, stderr = self.current_process.communicate(timeout=1)
                        if stdout:
                            print(f"程序输出: {stdout}")
                        if stderr:
                            print(f"程序错误: {stderr}")
                    except subprocess.TimeoutExpired:
                        pass
                    
                    # 如果是语音程序退出，自动重启预测程序
                    if self.current_program == 'voice':
                        print("🔄 语音程序结束，重启预测程序...")
                        time.sleep(2)  # 等待内存释放
                        self.start_prediction_program()
                    
                    return False
                else:
                    # 程序仍在运行
                    return True
                    
            return True
            
        except Exception as e:
            print(f"⚠️ 监控程序状态失败: {e}")
            return False
    
    def monitor_voice_input(self):
        """监控语音输入"""
        if not self.audio_serial:
            print("❌ 语音模块未初始化，跳过语音监控")
            return
        
        print("🎤 开始监控语音输入...")
        
        # 首先启动预测程序
        if not self.start_prediction_program():
            print("❌ 初始启动预测程序失败")
            return
        
        while self.running:
            try:
                # 监控当前程序状态
                if not self.monitor_current_program():
                    continue
                
                # 只有在预测程序运行时才监控语音输入
                if self.current_program != 'prediction':
                    time.sleep(0.5)
                    continue
                
                # 读取串口数据
                data = self.audio_serial.read(100)
                if data:
                    try:
                        voice_text = data.decode('gbk').strip()
                        
                        if voice_text:
                            print(f"🎤 检测到语音输入: {voice_text}")
                            
                            # 启动语音处理程序
                            if self.start_voice_program(voice_text):
                                print("✅ 语音程序启动成功，等待处理完成...")
                                
                                # 等待语音程序完成
                                while self.running and self.current_program == 'voice':
                                    if not self.monitor_current_program():
                                        break
                                    time.sleep(1)
                            else:
                                print("❌ 语音程序启动失败，继续预测程序")
                                
                    except UnicodeDecodeError as e:
                        print(f"语音数据解码失败: {e}")
                        continue
                        
                time.sleep(0.1)
                
            except Exception as e:
                print(f"语音监控错误: {e}")
                time.sleep(1)
    
    def cleanup(self):
        """清理资源"""
        print("🔄 清理资源...")
        
        # 关闭LED
        if self.led_chip:
            try:
                self.set_led_color(0, 0, 0)  # 关闭LED
                print("💡 LED已关闭")
            except Exception as e:
                print(f"⚠️ LED关闭失败: {e}")
        
        # 关闭语音串口
        if self.audio_serial:
            try:
                self.audio_serial.close()
                print("✅ 语音串口已关闭")
            except Exception as e:
                print(f"⚠️ 语音串口关闭失败: {e}")
        
        # 终止当前程序
        self.kill_existing_programs()
        
        print("✅ 资源清理完成")
    
    def start(self):
        """启动语音监控系统"""
        print("🚀 启动语音监控系统...")
        
        # 设置信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # 初始化语音模块
        if not self.init_audio_serial():
            print("❌ 语音模块初始化失败")
            return False
        
        # 设置运行标志
        self.running = True
        
        # 开始监控
        self.monitor_voice_input()
        
        # 清理资源
        self.cleanup()
        
        return True

def main():
    """主函数"""
    print("=" * 60)
    print("ELF2语音监控系统 - 程序切换管理")
    print("=" * 60)
    
    # 创建语音监控器实例
    monitor = VoiceMonitor(audio_port='/dev/ttyS9')
    
    print("\n🎯 程序功能:")
    print("- 持续监控语音输入")
    print("- 自动管理程序切换")
    print("- 内存优化和进程管理")
    print("- 确保程序间完全隔离")
    print()
    print("🔧 工作流程:")
    print("1. 启动交通预测程序")
    print("2. 监控语音输入")
    print("3. 检测到语音时切换到语音程序")
    print("4. 语音处理完成后自动切换回预测程序")
    print("5. 循环执行上述流程")
    print()
    print("💾 内存管理:")
    print("- 程序切换时完全终止旧进程")
    print("- 强制垃圾回收释放内存")
    print("- 实时监控内存状态")
    print("- 确保内存完全释放")
    print()
    print("📡 进程通信:")
    print("- 语音输入文件: /tmp/voice_input.txt")
    print("- 交通状态文件: /tmp/traffic_status.json")
    print("- 语音触发文件: /tmp/voice_trigger.txt")
    print()
    print("按Ctrl+C退出程序")
    print("=" * 60)
    
    try:
        # 启动系统
        monitor.start()
    except KeyboardInterrupt:
        print("\n接收到中断信号")
    except Exception as e:
        print(f"程序运行错误: {e}")
    finally:
        print("程序结束")

if __name__ == "__main__":
    main() 