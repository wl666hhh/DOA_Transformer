import torch

# --- 数据生成参数 ---
NUM_ARRAY_ELEMENTS = 10  # M: 均匀线性阵列（ULA）中的天线元件数量
SNAPSHOTS = 256          # L: 快照数
ARRAY_SPACING = 0.5      # 天线间距（单位：波长 d/lambda）

# --- 角度和网格参数 ---
ANGLE_MIN = -40          # 最小角度（度）
ANGLE_MAX = 40           # 最大角度（度）
ANGLE_RESOLUTION = 1     # 角度分辨率（度）
GRID_SIZE = int((ANGLE_MAX - ANGLE_MIN) / ANGLE_RESOLUTION) + 1  # 角度网格点总数

# --- Transformer解码器词汇表定义 ---
# 将角度索引、填充符、起始符、结束符统一管理
PAD_TOKEN = GRID_SIZE        # 填充符
START_TOKEN = GRID_SIZE + 1  # 起始符
END_TOKEN = GRID_SIZE + 2    # 结束符
VOCAB_SIZE = GRID_SIZE + 3   # 词汇表示总大小

# --- Transformer模型参数 (来自论文表4.1) ---
D_MODEL = 216            # 模型维度
N_HEAD = 6               # 多头注意力机制的头数
NUM_ENCODER_LAYERS = 3   # 编码器层数
NUM_DECODER_LAYERS = 3   # 解码器层数
D_FF = 1024              # 前馈神经网络的内部维度
DROPOUT = 0.3            # Dropout比例
MAX_SOURCES = 3          # 模型能检测的最大信源数
MAX_SEQ_LENGTH = MAX_SOURCES + 2 # 最大序列长度 = 最大信源数 + 起始/结束符

# --- 训练参数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # 自动选择GPU或CPU
LEARNING_RATE = 0.0001   # 学习率
BATCH_SIZE = 128         # 批处理大小
NUM_EPOCHS = 50          # 训练轮数（可根据收敛情况调整）
NUM_TRAIN_SAMPLES = 25600# 训练样本数量
NUM_VAL_SAMPLES = 2560   # 验证样本数量

# --- 文件路径 ---
TRAIN_DATA_PATH = "data/train_data.npz"      # 训练数据路径
VAL_DATA_PATH = "data/val_data.npz"        # 验证数据路径
MODEL_PATH = "saved_models/transformer_doa_model.pth" # 模型保存路径