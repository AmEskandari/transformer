import torch
import torch.nn.functional as F
import torch.nn as nn

from math import sqrt, sin, cos

# Self-Attention Layer is done!
class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()

        self.W_q = nn.Linear(input_dim, hidden_dim, bias= False)
        self.W_k = nn.Linear(input_dim, hidden_dim, bias= False)
        self.W_v = nn.Linear(input_dim, hidden_dim, bias= False)

    def forward(self, x, h=None):

        if isinstance(h,torch.Tensor):
            query = self.W_q(x)
            key = self.W_k(h)
            value = self.W_v(h)
        else:
            query = self.W_q(x)
            key = self.W_k(x)
            value = self.W_v(x)
        
        attention_weights = F.softmax((query @ key.transpose(-2, -1)) / (key.size(-1) ** 0.5), dim=-1)
        attended_values = attention_weights @ value

        return attended_values


# Add and Norm
class AddAndNormLayer(torch.nn.Module):
    def __init__(self, input_dim):
        super(AddAndNormLayer, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(input_dim)

    def forward(self, x, sub_layer_output):
        
        added = x + sub_layer_output
        normalized = self.layer_norm(added)

        return normalized

        
# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, heads):
        super(MultiHeadAttention, self).__init__()

        self.input_dim = input_dim
        self.heads = heads
        self.head_dim = input_dim // heads
        
        assert (
            self.head_dim * heads == input_dim
        )

        self.attention_heads = nn.ModuleList([
            SelfAttention(input_dim, self.head_dim) for _ in range(self.heads)
            ])
        
        self.fc_out = torch.nn.Linear(self.heads * self.head_dim, self.input_dim)
        self.add_and_norm = AddAndNormLayer(self.head_dim * self.heads)       
        
    def forward(self, x, h = None):

        if isinstance(h,torch.Tensor):
            head_outputs = [attention_head(x,h) for attention_head in self.attention_heads]
            concatenated_output = self.fc_out(torch.cat(head_outputs, dim=-1)) 

        else:
            head_outputs = [attention_head(x) for attention_head in self.attention_heads]
            concatenated_output = self.fc_out(torch.cat(head_outputs, dim=-1)) 
            
        return self.add_and_norm(concatenated_output,x)



# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model, n =10000, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe[:seq_len, :]

    def forward(self, x):
        return x + self.pe


# Position-wise Feed-Forward Networks
class PositionWiseFFN(nn.Module):
    def __init__(self, input_dim,hidden_dim):
        super(PositionWiseFFN,self).__init__()

        self.fc_1 = nn.Linear(input_dim,hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim,input_dim)
        self.add_and_norm = AddAndNormLayer(input_dim)

    def forward(self,x):
        h = F.relu(self.fc_1(x))
        h = self.fc_2(h)
        return self.add_and_norm(h,x)


class Encoder(nn.Module):
    def __init__(self,input_dim, hidden_dim, heads):
        super(Encoder,self).__init__()
        self.att_block = MultiHeadAttention(input_dim, heads)   
        self.fnn_block = PositionWiseFFN(input_dim,hidden_dim)

    def forward(self,x):
        h = self.fnn_block(x)
        h = self.att_block(h)
        return h


class Decoder(nn.Module):
    def __init__(self,input_dim, hidden_dim, heads):
        super(Decoder,self).__init__()
        self.masked_att_block = MultiHeadAttention(input_dim,heads)
        self.att_block = MultiHeadAttention(input_dim,heads)
        self.fnn_block = PositionWiseFFN(input_dim,hidden_dim)

    def forward(self,x,x_enc):
        h = self.masked_att_block(x)
        h = self.att_block(h,x_enc)
        x = self.fnn_block(h)
        return h


class Prediction(nn.Module):
    def __init__(self,input_dim,vocab_size):
        super(Prediction,self).__init__()
        self.clf = nn.Linear(input_dim,vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self,x):
        return self.softmax(self.clf(x))

#TODO: Transformer Class
#TODO: Assert for wrong data type and inpute type.



