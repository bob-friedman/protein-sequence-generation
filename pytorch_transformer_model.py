import math
import torch
import torch.nn as nn
import torch.nn.functional as F # Not strictly required by prompt but often useful

class PositionalEncoding(nn.Module):
    """
    Injects sinusoidal positional encoding into the input tensor.

    The positional encodings have the same dimension as the embeddings so that
    the two can be summed. Here, we use sine and cosine functions of different
    frequencies.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    where pos is the position and i is the dimension.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model: The dimension of the embeddings (and the model).
            dropout: The dropout probability.
            max_len: The maximum length of the input sequences.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # (d_model/2)
        
        pe = torch.zeros(max_len, 1, d_model) # (max_len, 1, d_model) for broadcasting with (seq_len, batch, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # register_buffer ensures that 'pe' is part of the model's state_dict,
        # but not updated by the optimizer during training.
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape ``[seq_len, batch_size, d_model]``

        Returns:
            Tensor with positional encoding added, shape ``[seq_len, batch_size, d_model]``
        """
        # x.size(0) is seq_len. We add the positional encoding up to that length.
        # self.pe is (max_len, 1, d_model), self.pe[:x.size(0)] is (seq_len, 1, d_model)
        # This will broadcast correctly with x of shape (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ProteinTransformerModel(nn.Module):
    """
    A decoder-only Transformer model for protein sequence generation (causal language modeling).
    """

    def __init__(self, vocab_size: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            vocab_size: The size of the vocabulary (number of unique amino acids/tokens).
            d_model: The dimension of the embeddings and the model's internal states.
            nhead: The number of attention heads in the multiheadattention models.
            d_hid: The dimension of the feedforward network model in nn.TransformerDecoderLayer.
            nlayers: The number of nn.TransformerDecoderLayer layers in the nn.TransformerDecoder.
            dropout: The dropout probability (applied in PositionalEncoding and TransformerDecoder).
            max_len: The maximum sequence length for positional encoding.
        """
        super().__init__()
        self.model_type = 'TransformerDecoder'
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, d_hid, dropout, batch_first=False)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, nlayers)
        
        self.output_layer = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initializes weights of the linear and embedding layers.
        """
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)

    @staticmethod
    def _generate_square_subsequent_mask(sz: int, device: torch.device = None) -> torch.Tensor:
        """
        Generates a square mask for causal attention.

        For a target sequence of length ``sz``, the element ``mask[i, j]`` is ``0.``
        if ``j <= i`` (attention allowed) and ``-inf`` if ``j > i`` (attention prevented).

        Args:
            sz: The size of the sequence length dimension.
            device: The device to create the mask on.

        Returns:
            A tensor of shape ``(sz, sz)``.
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the ProteinTransformerModel.

        Args:
            src: The input sequence of token IDs, shape ``(seq_len, batch_size)``.
            src_mask: An optional square causal mask, shape ``(seq_len, seq_len)``.
                      If None, a default causal mask will be generated.

        Returns:
            The output logits from the model, shape ``(seq_len, batch_size, vocab_size)``.
        """
        seq_len = src.size(0)
        device = src.device

        if src_mask is None:
            # Note: The nn.TransformerDecoder expects the mask to be (L, S) for target_mask
            # which in our decoder-only setup means (seq_len, seq_len).
            src_mask = self._generate_square_subsequent_mask(seq_len, device=device)
        
        src_embedded = self.embedding(src) * math.sqrt(self.d_model)
        src_pos_encoded = self.pos_encoder(src_embedded)
        
        # TransformerDecoder expects inputs of shape (target_seq_len, batch_size, embed_dim)
        # and memory (source_seq_len, batch_size, embed_dim).
        # In a decoder-only setup, memory is effectively the same as the target sequence.
        # The target_mask (src_mask here) prevents attending to future positions in the target.
        # There is no memory_mask needed as we are not using distinct memory.
        output = self.transformer_decoder(tgt=src_pos_encoded, memory=src_pos_encoded, 
                                          tgt_mask=src_mask, memory_mask=src_mask)
        
        logits = self.output_layer(output)
        return logits

if __name__ == '__main__':
    # Example Usage:
    vocab_size = 25  # Example: 20 amino acids + special tokens
    d_model = 512
    nhead = 8
    d_hid = 2048
    nlayers = 6
    dropout = 0.1
    max_len_example = 100 # Shorter for example

    model = ProteinTransformerModel(vocab_size, d_model, nhead, d_hid, nlayers, dropout, max_len_example)
    model.eval() # Set to evaluation mode for example inference

    # Example input: batch of 2 sequences, each of length 10
    # Shape: (seq_len, batch_size)
    example_src = torch.randint(0, vocab_size, (10, 2)) 
    
    # For a decoder-only model, memory_mask is often the same as tgt_mask if memory is src_pos_encoded itself.
    # The nn.TransformerDecoder is flexible. If memory is not provided, it won't use it.
    # However, the API requires memory. So we pass src_pos_encoded as memory.
    # The crucial part is the tgt_mask for causal attention on the target sequence.
    
    # Generate a causal mask for the example sequence length
    example_seq_len = example_src.size(0)
    # causal_mask = ProteinTransformerModel._generate_square_subsequent_mask(example_seq_len)

    # Get logits (no mask needed for this call if model generates it internally)
    # If you want to pass it explicitly:
    # logits = model(example_src, src_mask=causal_mask)
    logits = model(example_src) # Model will generate mask if None

    print("Input shape:", example_src.shape)
    print("Output logits shape:", logits.shape) # Expected: (10, 2, vocab_size)

    # Test positional encoding standalone
    pe = PositionalEncoding(d_model=d_model, max_len=max_len_example)
    test_tensor = torch.zeros(10, 2, d_model) # (seq_len, batch, d_model)
    encoded_tensor = pe(test_tensor)
    print("PositionalEncoding output shape:", encoded_tensor.shape) # Expected: (10, 2, d_model)

    # Test mask generation
    mask = ProteinTransformerModel._generate_square_subsequent_mask(5)
    print("Generated causal mask (5x5):\n", mask)
    # Expected:
    # tensor([[0., -inf, -inf, -inf, -inf],
    #         [0., 0., -inf, -inf, -inf],
    #         [0., 0., 0., -inf, -inf],
    #         [0., 0., 0., 0., -inf],
    #         [0., 0., 0., 0., 0.]])
