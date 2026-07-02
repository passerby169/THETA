import sys
import os

# 将 theta_project 目录加入 Python 模块搜索路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'theta_project'))

# 从 theta_project/asgi.py 中导入你的 FastAPI 实例 app
from asgi import app
