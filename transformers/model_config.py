import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from dataclasses import dataclass

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        # This is from the paper where we multiply the embeddings by sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create position embeddings
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Fill position embedding matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        # when model is saved, this will be saved
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, last_dim: int, epsilon: float=1e-6):
        super().__init__()
        self.last_dim = last_dim
        self.alpha = nn.Parameter(torch.ones(last_dim), requires_grad=True) 
        self.beta = nn.Parameter(torch.zeros(last_dim), requires_grad=True)
        self.epsilon = epsilon
    
    def forward(self, x: torch.Tensor):
        xmean = x.mean(dim=-1, keepdim=True)
        xvar = x.var(dim=-1, keepdim=True)
        xnorm = (x - xmean) / torch.sqrt(xvar + self.epsilon)
        return self.alpha * xnorm + self.beta

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float): 
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1, b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2, b2 
    
    def forward(self, x: torch.Tensor):
        x = F.relu(self.linear_1(x))
        return self.linear_2(self.dropout(x))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h # number of heads
        assert d_model % h == 0, "d_model it not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # w_q
        self.w_k = nn.Linear(d_model, d_model, bias=False) # w_k
        self.w_v = nn.Linear(d_model, d_model, bias=False) # w_v
        self.w_o = nn.Linear(d_model, d_model, bias=False) # w_o

        self.dropout = nn.Dropout(dropout)
    @staticmethod
    def attention(q, k, v, mask, dropout: nn.Dropout):
        # q, k, v are of dim (batch, h, seq_len, d_k)
        d_k = q.shape[-1]

        # (batch, h, seq_len, seq_len)
        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_scores = attention_scores.softmax(dim=-1) 
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # (batch, h, seq_len, d_k), (batch, h, seq_len, seq_len)
        return (attention_scores @ v), attention_scores
    
    def forward(self, q, k, v, mask):
        # (batch, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # get attention
        x, attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        # (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (batch, seq_len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, features: int, encoder_blocks: nn.ModuleList):
        super().__init__()
        self.encoder_blocks = encoder_blocks
        self.norm = LayerNormalization(features)
    
    def forward(self, x, src_mask):
        for encder_block in self.encoder_blocks:
            x = encder_block(x, src_mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)]
        )
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, features: int, decoder_blocks: nn.ModuleList):
        super().__init__()
        self.norm = LayerNormalization(features)
        self.decoder_blocks = decoder_blocks
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_output, src_mask, tgt_mask)
        
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionEncoding, tgt_pos: PositionEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)

@dataclass
class ModelArgs:
    src_vocab_size: int
    tgt_vocab_size: int
    src_seq_len: int
    tgt_seq_len: int
    d_model: int = 512
    N: int = 6 # number of encoder / decoder blocks
    h: int = 8 # number of heads in multi head attention
    dropout: float = 0.1
    d_ff: int = 2048

def build_transformer(model_args: ModelArgs):
    # Create embedding vectors
    d_model = model_args.d_model
    src_embed = InputEmbeddings(d_model, model_args.src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, model_args.tgt_vocab_size)

    # Create position embeddings
    src_pos = PositionEncoding(d_model, model_args.src_seq_len, model_args.dropout)
    tgt_pos = PositionEncoding(d_model, model_args.tgt_seq_len, model_args.dropout)
    
    # Create encoder blocks
    N = model_args.N
    encoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, model_args.h, model_args.dropout)
        feed_forward_block = FeedForwardBlock(d_model, model_args.d_ff, model_args.dropout)
        encoder_block = EncoderBlock(
            d_model,
            self_attention_block,
            feed_forward_block,
            model_args.dropout
        )
        encoder_blocks.append(encoder_block)
    
    # Create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, model_args.h, model_args.dropout)
        cross_attention_block = MultiHeadAttentionBlock(d_model, model_args.h, model_args.dropout)
        feed_forward_block = FeedForwardBlock(d_model, model_args.d_ff, model_args.dropout)
        decoder_block = DecoderBlock(
            d_model,
            self_attention_block,
            cross_attention_block,
            feed_forward_block,
            model_args.dropout
        )
        decoder_blocks.append(decoder_block)
    
    # Create encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, model_args.tgt_vocab_size)

    transformer = Transformer(
        encoder,
        decoder,
        src_embed,
        tgt_embed,
        src_pos,
        tgt_pos,
        projection_layer
    )

    # Initialize the parameters (xavier init)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer