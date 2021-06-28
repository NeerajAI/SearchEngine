class Dataset:
    def __init__(data):
        self.rawdata = data
        self.s = data
    def unicode_to_ascii(self):
        return ''.join(
        c for c in unicodedata.normalize('NFD', self.s)
        if unicodedata.category(c) != 'Mn')
    @classmethod
    def normalize_string(cls):
        cls.s = unicode_to_ascii(cls.s)
        cls.s = re.sub(r'([!.?])', r' \1', cls.s)
        cls.s = re.sub(r'[^a-zA-Z.!?]+', r' ', cls.s)
        cls.s = re.sub(r'\s+', r' ', s)
        return cls.s
    
    def basic_operation(self):
        raw_data_en, raw_data_fr = list(zip(*self.rawdata))
        raw_data_en,raw_data_fr = list(raw_data_en), list(raw_data_fr)
        raw_data_en = [normalize_string(data) for data in raw_data_en]
        raw_data_fr_in = ['<start> ' + normalize_string(data) for data in raw_data_fr]
        raw_data_fr_out = [normalize_string(data) + ' <end>' for data in raw_data_fr]

class MultiHeadAttention(tf.keras.Model):
    def __init__(self, model_size, h):
        super(MultiHeadAttention, self).__init__()
        self.query_size = model_size // h
        self.key_size = model_size // h
        self.value_size = model_size // h
        self.h = h
        self.wq = [tf.keras.layers.Dense(self.query_size) for _ in range(h)]
        self.wk = [tf.keras.layers.Dense(self.key_size) for _ in range(h)]
        self.wv = [tf.keras.layers.Dense(self.value_size) for _ in range(h)]
        self.wo = tf.keras.layers.Dense(model_size)

    def call(self, decoder_output, encoder_output):
        # decoder_output has shape (batch, decoder_len, model_size)
        # encoder_output has shape (batch, encoder_len, model_size)
        heads = []
        for i in range(self.h):
            score = tf.matmul(self.wq[i](decoder_output), self.wk[i](encoder_output), transpose_b=True) / tf.math.sqrt(tf.dtypes.cast(self.key_size, tf.float32))
            # score has shape (batch, decoder_len, encoder_len)
            alignment = tf.nn.softmax(score, axis=2)
            # alignment has shape (batch, decoder_len, encoder_len)
            head = tf.matmul(alignment, self.wv[i](encoder_output))
            # head has shape (batch, decoder_len, value_size)
            heads.append(head)
        heads = tf.concat(heads, axis=2)
        heads = self.wo(heads)
        # heads has shape (batch, decoder_len, model_size)
        return heads