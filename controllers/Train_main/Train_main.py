import sys
import os
# 添加项目路径到系统路径
_conda = r"D:\\Anaconda3\\envs\\webots"
_extra = [
    _conda,
    rf"{_conda}\Library\mingw-w64\bin",
    rf"{_conda}\Library\usr\bin",
    rf"{_conda}\Library\bin",
    rf"{_conda}\Scripts",
    rf"{_conda}\bin",
]
os.environ["PATH"] = ";".join(_extra) + ";" + os.environ.get("PATH", "")

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CUR_DIR, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from python_scripts.DQN import DQN_episoid
from python_scripts.PPO import PPO_episoid_1
from python_scripts.SAC import SAC_episoid
from python_scripts.Project_config import path_list

def main():
    # 直接指定模型路径
    #model_path = "D:/project/python_scripts/DQN/checkpoint/dqn_model_0.ckpt"
    
    #print("将使用DQN进行训练")
    #DQN_episoid.DQN_episoid()#model_path=model_path

    print("将使用PPO进行训练")
    PPO_episoid_1.PPO_episoid_1()

    #print("将使用SAC进行训练")
    #SAC_episoid.SAC_episoid()
    
if __name__ == '__main__':
    print("_________")
    with open(path_list['resetFlag'], 'r+') as file:
        file.write('1')
    main()
