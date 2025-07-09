#!/usr/bin/env python3
"""
ELF2智能交通系统启动脚本
"""

import os
import sys
import subprocess
import signal
import time

def check_dependencies():
    """检查依赖文件是否存在"""
    print("🔍 检查依赖文件...")
    
    required_files = [
        '/userdata/deepseek/python/voice_monitor_offline.py',
        '/userdata/deepseek/python/traffic_prediction_main_offline.py',
        '/userdata/deepseek/python/voice_processing_main_offline.py',
        '/userdata/deepseek/python/map.txt',
        '/userdata/deepseek/python/traffic_model_rk3588.rknn'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print("❌ 以下文件缺失:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print("✅ 所有依赖文件检查完成")
    return True

def check_hardware():
    """检查硬件组件"""
    print("🔍 检查硬件组件...")
    
    # 检查音频模块
    audio_ok = False
    try:
        import serial
        audio_serial = serial.Serial('/dev/ttyS9', 9600, timeout=1)
        audio_serial.close()
        audio_ok = True
        print("✅ 音频模块 (/dev/ttyS9)")
    except Exception as e:
        print(f"❌ 音频模块 (/dev/ttyS9): {e}")
    
    # 检查摄像头
    camera_ok = False
    try:
        import cv2
        cap = cv2.VideoCapture(21)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                camera_ok = True
                print("✅ 摄像头 (端口 21)")
            else:
                print("❌ 摄像头 (端口 21): 无法读取帧")
        else:
            print("❌ 摄像头 (端口 21): 无法打开")
        cap.release()
    except Exception as e:
        print(f"❌ 摄像头 (端口 21): {e}")
    
    # 检查LED控制
    led_ok = False
    try:
        import gpiod
        chip = gpiod.Chip('gpiochip3')
        led_red = chip.get_line(4)
        led_green = chip.get_line(0)
        led_blue = chip.get_line(3)
        led_ok = True
        print("✅ LED控制 (GPIO 4,0,3)")
    except Exception as e:
        print(f"❌ LED控制: {e}")
    
    return audio_ok and camera_ok and led_ok

def cleanup_temp_files():
    """清理临时文件"""
    print("🔄 清理临时文件...")
    
    temp_files = [
        '/tmp/voice_input.txt',
        '/tmp/traffic_status.json',
        '/tmp/voice_trigger.txt',
        '/tmp/traffic_command.txt'
    ]
    
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"✅ 已清理: {file_path}")
        except Exception as e:
            print(f"⚠️ 清理失败 {file_path}: {e}")

def signal_handler(signum, frame):
    """处理系统信号"""
    print(f"\n接收到信号 {signum}，正在退出...")
    cleanup_temp_files()
    sys.exit(0)

def main():
    """主函数"""
    print("=" * 60)
    print("ELF2智能交通系统启动器")
    print("=" * 60)
    
    print("🚀 启动系统初始化...")
    
    # 检查依赖
    if not check_dependencies():
        print("❌ 依赖检查失败，请检查文件路径")
        return False
    
    # 检查硬件
    if not check_hardware():
        print("❌ 硬件检查失败，请检查硬件连接")
        return False
    
    # 清理临时文件
    cleanup_temp_files()
    
    print("\n🎯 系统架构:")
    print("- 语音监控器: 管理程序切换")
    print("- 交通预测程序: 实时预测交通状况")
    print("- 语音处理程序: 智能语音交互")
    print("- 进程间通信: 文件和信号管理")
    print("- 摄像头端口: 21")
    print("- 工作目录: /userdata/deepseek/python")
    print()
    
    try:
        print("🚀 启动语音监控系统...")
        import subprocess
        
        # 切换到工作目录
        os.chdir('/userdata/deepseek/python')
        
        # 启动语音监控器
        cmd = ['python3', 'voice_monitor_offline.py']
        process = subprocess.Popen(cmd)
        
        print("✅ 语音监控系统已启动")
        print("按Ctrl+C停止系统")
        
        # 等待进程结束
        process.wait()
        
    except KeyboardInterrupt:
        print("\n接收到中断信号，正在停止系统...")
        try:
            process.terminate()
            process.wait(timeout=10)
        except:
            process.kill()
    except Exception as e:
        print(f"启动系统失败: {e}")
    finally:
        cleanup_temp_files()
        print("系统已停止")
    
    return True

if __name__ == "__main__":
    main() 
