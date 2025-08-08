class RNNConfig:
    def __init__(self):
        self.vocab_size=52000
        self.hidden_dim=64
        self.seq_len = 512
        self.batch_size=32
        self.epochs=100
        self.learning_rate=1e-4
        self.weight_decay=0.0001
        self.save_pth='./tmp'
        self.log_dir = './log'
        self.pad_token_id=None
        self.eos_token_id=None
        self.scheduler_step = 500
        self.use_amp=False
        self.dataset_pth="./wmt_dataset_de_en_tokenized.parquet"
        self.pretrained_pth=None
        
        