import torch
import torch.nn as nn
import numpy as np
import math
import config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class DOATransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, dropout, vocab_size):
        super(DOATransformer, self).__init__()

        self.input_proj = nn.Linear(2, d_model)  # Project [real, imag] to d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=config.NUM_ARRAY_ELEMENTS ** 2)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_len=config.MAX_SEQ_LENGTH)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        # src: [batch, seq_len, 2] (seq_len = M*M)
        # tgt: [batch, tgt_len]

        # Encoder Path
        src_proj = self.input_proj(src) * math.sqrt(config.D_MODEL)
        src_pos = self.pos_encoder(src_proj.transpose(0, 1)).transpose(0, 1)  # Transformer expects seq_len first for PE

        # Decoder Path
        tgt_emb = self.embedding(tgt) * math.sqrt(config.D_MODEL)
        tgt_pos = self.pos_decoder(tgt_emb.transpose(0, 1)).transpose(0, 1)
        #生成的 tgt_mask 形状为 [tgt_len, tgt_len] 是上三角矩阵
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(config.DEVICE)

        output = self.transformer(src_pos, tgt_pos, tgt_mask=tgt_mask)

        return self.fc_out(output)


if __name__ == '__main__':
    # Test the model
    model = DOATransformer(
        d_model=config.D_MODEL,
        nhead=config.N_HEAD,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        dim_feedforward=config.D_FF,
        dropout=config.DROPOUT,
        vocab_size=config.VOCAB_SIZE
    ).to(config.DEVICE)

    # 演示
    src = torch.rand(config.BATCH_SIZE, config.NUM_ARRAY_ELEMENTS ** 2, 2).to(config.DEVICE)
    tgt = torch.randint(0, config.VOCAB_SIZE, (config.BATCH_SIZE, config.MAX_SEQ_LENGTH)).to(config.DEVICE)

    out = model(src, tgt[:, :-1])  # Use teacher forcing, exclude last token from input
    print("Output shape:", out.shape)  # Should be [batch_size, max_seq_len-1, vocab_size]