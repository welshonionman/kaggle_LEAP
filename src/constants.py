import torch

DATA_PATH = "/kaggle/input/"
SAMPLE_PATH = "/kaggle/input/leap-atmospheric-physics-ai-climsim/sample_submission.csv"
TRAIN_PATH = "/kaggle/input/leap-atmospheric-physics-ai-climsim/train.csv"
TEST_PATH = "/kaggle/input/leap-atmospheric-physics-ai-climsim/test.csv"
SEQ_FEATURES = [
    "state_t",
    "state_q0001",
    "state_q0002",
    "state_q0003",
    "state_u",
    "state_v",
    "pbuf_ozone",
    "pbuf_CH4",
    "pbuf_N2O",
]

SINGLE_FEATURES = [
    "state_ps",
    "pbuf_SOLIN",
    "pbuf_LHFLX",
    "pbuf_SHFLX",
    "pbuf_TAUX",
    "pbuf_TAUY",
    "pbuf_COSZRS",
    "cam_in_ALDIF",
    "cam_in_ALDIR",
    "cam_in_ASDIF",
    "cam_in_ASDIR",
    "cam_in_LWUP",
    "cam_in_ICEFRAC",
    "cam_in_LANDFRAC",
    "cam_in_OCNFRAC",
    "cam_in_SNOWHLAND",
]

SEQ_TARGETS = [
    "ptend_t",
    "ptend_q0001",
    "ptend_q0002",
    "ptend_q0003",
    "ptend_u",
    "ptend_v",
]

SINGLE_TARGETS = [
    "cam_out_NETSW",
    "cam_out_FLWDS",
    "cam_out_PRECSC",
    "cam_out_PRECC",
    "cam_out_SOLS",
    "cam_out_SOLL",
    "cam_out_SOLSD",
    "cam_out_SOLLD",
]

SEQ_LEN = 60

SEQ_FEATURES_COL_LEN = len(SEQ_FEATURES)  # 9
SINGLE_FEATURES_COL_LEN = len(SINGLE_FEATURES)  # 16
SEQ_TARGETS_COL_LEN = len(SEQ_TARGETS)  # 6
SINGLE_TARGETS_COL_LEN = len(SINGLE_TARGETS)  # 8

FEAT_COLS = [f"{feature}_{i}" for feature in SEQ_FEATURES for i in range(SEQ_LEN)] + SINGLE_FEATURES  # 556
TARGET_COLS = [f"{target}_{i}" for target in SEQ_TARGETS for i in range(SEQ_LEN)] + SINGLE_TARGETS  # 368


FEAT_LEN = len(FEAT_COLS)  # 556
TARGET_LEN = len(TARGET_COLS)  # 368

SEQ_FEATURES_IDX_1 = slice(0, SEQ_LEN * 6)  # (0, 360)
SINGLE_FEATURES_IDX = slice(SEQ_LEN * 6, SEQ_LEN * 6 + 16)  # (360, 376)
SEQ_FEATURES_IDX_2 = slice(SEQ_LEN * 6 + 16, SEQ_LEN * 9 + 16)  # (376, 556)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
