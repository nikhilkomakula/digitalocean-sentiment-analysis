# declare constants
DATASET_FILENAME = 'training.1600000.processed.noemoticon.csv'
DATASET_COLUMNS = ['sentiment', 'id', 'timestamp', 'flag', 'user', 'text']
TEXT_PREPROCESS_RE = '@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+'
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 30
W2V_MIN_COUNT = 10
MAX_LENGTH = 300
POSITIVE = 'Positive'
NEGATIVE = 'Negative'
NEUTRAL = 'Neutral'
SENTIMENT_THRESHOLD = 0.5
SENTIMENT_THRESHOLDS = (0.4, 0.6)
SA_MODEL = 'sa_model.keras'
TOKENIZER = 'tokenizer.pkl'
MODELS_FOLDER = 'models'