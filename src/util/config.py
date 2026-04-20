import os

def _load_env(env_file=None):
    """
    从 .env 文件加载配置到 os.environ（不依赖 python-dotenv）。
    优先级：已有的环境变量 > .env 文件 > 默认值。
    """
    if env_file is None:
        # 向上查找项目根目录的 .env
        cur = os.path.dirname(os.path.abspath(__file__))
        for _ in range(4):
            candidate = os.path.join(cur, '.env')
            if os.path.exists(candidate):
                env_file = candidate
                break
            cur = os.path.dirname(cur)

    if env_file and os.path.exists(env_file):
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip()
                # 只在未被外部环境变量覆盖时才设置
                if key and key not in os.environ:
                    os.environ[key] = value

_load_env()

# -------- 数据集路径 --------
CUB_DATA_DIR    = os.environ.get('CUB_DATA_DIR',    'src/data/CUB_200_2011')
AWA2_DATA_DIR   = os.environ.get('AWA2_DATA_DIR',   'src/data/AwA2')
TIL_DATA_DIR    = os.environ.get('TIL_DATA_DIR',    'src/data/TIL')
ANIMALS_DATA_DIR= os.environ.get('ANIMALS_DATA_DIR','src/data/animals')
CELEBA_DATA_DIR = os.environ.get('CELEBA_DATA_DIR', 'src/data/CelebA')
OOD_BIRD_DATA_DIR = os.environ.get('OOD_BIRD_DATA_DIR', 'src/data/OOD_birds')

# -------- 输出路径 --------
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'outputfiles')

# -------- 旧版兼容（BASE_DIR 原为空字符串） --------
BASE_DIR = os.environ.get('BASE_DIR', '')

# -------- 数据集参数 --------
N_ATTRIBUTES = int(os.environ.get('N_ATTRIBUTES', '312'))
N_CLASSES    = int(os.environ.get('N_CLASSES',    '200'))

# -------- 训练超参（不从 .env 读取，保持代码内定义） --------
UPWEIGHT_RATIO = 9.0
MIN_LR         = 0.0001
LR_DECAY_SIZE  = 0.1
