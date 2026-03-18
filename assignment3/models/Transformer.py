"""
Transformer model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np

import torch
from torch import nn
import random

####### Do not modify these imports.

class TransformerTranslator(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and 
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
        """
        :param input_size: the size of the input, which equals to the number of words in source language vocabulary
        :param output_size: the size of the output, which equals to the number of words in target language vocabulary
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        """
        super(TransformerTranslator, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        
        seed_torch(0)
        
        ##############################################################################
        # TODO:
        # Deliverable 1: Initialize what you need for the embedding lookup.          #
        # You will need to use the max_length parameter above.                       #
        # Don’t worry about sine/cosine encodings- use positional encodings.         #
        ##############################################################################
        self.embeddingL = nn.Embedding(self.input_size, self.hidden_dim)      #initialize word embedding layer
        self.posembeddingL = nn.Embedding(self.max_length, self.hidden_dim)   #initialize positional embedding layer

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        
        
        ##############################################################################
        # Deliverable 2: Initializations for multi-head self-attention.              #
        # You don't need to do anything here. Do not modify this code.               #
        ##############################################################################
        
        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q)
        
        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q)
        
        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)

        
        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the feed-forward layer.        # 
        # Don't forget the layer normalization.                                      #
        ##############################################################################
        self.ff_layer1 = nn.Linear(self.hidden_dim, self.dim_feedforward)  # First linear layer of the feed-forward network
        self.ff_layer2 = nn.Linear(self.dim_feedforward, self.hidden_dim)  # Second linear layer of the feed-forward network
        self.norm_ff = nn.LayerNorm(self.hidden_dim)  # Layer normalization for the feed 
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
        ##############################################################################
        # TODO:
        # Deliverable 4: Initialize what you need for the final layer (1-2 lines).   #
        ##############################################################################
        self.final_projection = nn.Linear(self.hidden_dim, self.output_size)  # Final linear layer to project to output size 
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
    def forward(self, inputs):
        """
        This function computes the full Transformer forward pass.
        Put together all of the layers you've developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups.

        :returns: the model outputs. Should be scores of shape (N,T,output_size).
        """

        #############################################################################
        # TODO:
        # Deliverable 5: Implement the full Transformer stack for the forward pass. #
        # You will need to use all of the methods you have previously defined above.#
        # You should only be calling TransformerTranslator class methods here.      #
        #############################################################################
        embeddings = self.embed(inputs)  # Get embeddings for the input sequences
        mh_outputs = self.multi_head_attention(embeddings)  # Apply multi-head attention
        ff_outputs = self.feedforward_layer(mh_outputs)  # Apply feedforward layer
        outputs = self.final_layer(ff_outputs)  # Apply final layer to get output scores
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
        #############################################################################
        # TODO:                                                                     #
        # Deliverable 1: Return the embeddings.                                     #
        # This will take a few lines.                                               #
        #############################################################################
        token_embeddings = self.embeddingL(inputs)    #get token embeddings from the embedding layer
        position_indices = torch.arange(inputs.size(1), device=inputs.device).unsqueeze(0)  #create position indices for positional embeddings
        position_embeddings = self.posembeddingL(position_indices)  #get positional embeddings from the positional embedding layer
        embeddings = token_embeddings + position_embeddings  #combine token and positional embeddings by summing them 
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return embeddings
        
    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """
        
        
        #############################################################################
        # TODO:
        # Deliverable 2: Implement multi-head self-attention followed by add + norm.#
        # Use the provided 'Deliverable 2' layers initialized in the constructor.   #
        #############################################################################
        Q1, K1, V1 = self.q1(inputs), self.k1(inputs), self.v1(inputs)  # Compute queries, keys, and values for head 1
        Q2, K2, V2 = self.q2(inputs), self.k2(inputs), self.v2(inputs)  # Compute queries, keys, and values for head 2
        # Compute attention scores
        A1 = torch.bmm(self.softmax(torch.bmm(Q1, K1.transpose(1, 2)) / (self.dim_k ** 0.5)), V1)  # Scaled dot-product attention for head 1
        A2 = torch.bmm(self.softmax(torch.bmm(Q2, K2.transpose(1, 2)) / (self.dim_k ** 0.5)), V2)  # Scaled dot-product attention for head 2
        # Concatenate attention outputs from both heads
        A = torch.cat((A1, A2), dim=2)  # Concatenate along the feature dimension
        # Project the concatenated attention outputs back to the hidden dimension
        projected_A = self.attention_head_projection(A)  # Linear projection of concatenated attention outputs
        # Add & Norm
        outputs = self.norm_mh(inputs + projected_A)  # Add & Norm step
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 3: Implement the feedforward layer followed by add + norm.    #
        # Use a ReLU activation and apply the linear layers in the order you        #
        # initialized them.                                                         #
        # This should not take more than 3-5 lines of code.                         #
        #############################################################################
        ff_output = self.ff_layer2(torch.relu(self.ff_layer1(inputs)))  # Feedforward network with ReLU activation
        outputs = self.norm_ff(inputs + ff_output)  # Add & Norm step, residual connection
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 4: Implement the final layer for the Transformer Translator.  #
        # This should only take about 1 line of code. Softmax is not needed here    #
        # as it is integrated as part of cross entropy loss function.               #
        #############################################################################
        outputs = self.final_projection(inputs)  # Project to output size
                
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        

class FullTransformerTranslator(nn.Module):

    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2,
                 dim_feedforward=2048, num_layers_enc=2, num_layers_dec=2, dropout=0.2, max_length=43, ignore_index=1):
        super(FullTransformerTranslator, self).__init__()

        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.pad_idx=ignore_index

        seed_torch(0)

        ##############################################################################
        # TODO:
        # Deliverable 1: Initialize what you need for the Transformer Layer          #
        # You should use nn.Transformer                                              #
        ##############################################################################
        self.transformer = nn.Transformer(
            d_model=self.hidden_dim, 
            nhead=self.num_heads, 
            num_encoder_layers=num_layers_enc, 
            num_decoder_layers=num_layers_dec, 
            dim_feedforward=self.dim_feedforward, 
            dropout=dropout,
            batch_first=True,
        )
        ##############################################################################
        # TODO:
        # Deliverable 2: Initialize what you need for the embedding lookup.          #
        # You will need to use the max_length parameter above.                       #
        # Initialize embeddings in order shown below.                                #
        # Don’t worry about sine/cosine encodings- use positional encodings.         #
        ##############################################################################
        # Do not change the order for these variables
        self.srcembeddingL = nn.Embedding(input_size, hidden_dim)       #embedding for src
        self.tgtembeddingL = nn.Embedding(output_size, hidden_dim)       #embedding for target
        self.srcposembeddingL = nn.Embedding(max_length, hidden_dim)    #embedding for src positional encoding
        self.tgtposembeddingL = nn.Embedding(max_length, hidden_dim)    #embedding for target positional encoding
        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the final layer.               #
        ##############################################################################
        self.final_projection = nn.Linear(self.hidden_dim, self.output_size)  # Final linear layer to project to output size
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, src, tgt):
        """
         This function computes the full Transformer forward pass used during training.
         Put together all of the layers you've developed in the correct order.

         :param src: a PyTorch tensor of shape (N,T) these are tokenized input sentences
                tgt: a PyTorch tensor of shape (N,T) these are tokenized translations
         :returns: the model outputs. Should be scores of shape (N,T,output_size).
         """
        #############################################################################
        # TODO:
        # Deliverable 4: Implement the full Transformer stack for the forward pass. #
        #############################################################################
        outputs=None
        # shift tgt to right, add one <sos> to the beginning and shift the other tokens to right
        tgt = self.add_start_token(tgt)


        # embed src and tgt for processing by transformer
        src_emb = self.srcembeddingL(src) + self.srcposembeddingL(torch.arange(src.size(1), device=src.device))
        tgt_emb = self.tgtembeddingL(tgt) + self.tgtposembeddingL(torch.arange(tgt.size(1), device=tgt.device))

        # create target mask and target key padding mask for decoder - Both have boolean values
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).bool().to(src.device)  # Mask to prevent attention to future tokens
        tgt_key_padding_mask = (tgt == self.pad_idx)  # Mask to ignore padding tokens in the target sequences

        # invoke transformer to generate output
        transformer_output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        # pass through final layer to generate outputs
        outputs = self.final_projection(transformer_output)  # Project to output size

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs

    def generate_translation(self, src):
        """
         This function generates the output of the transformer taking src as its input
         it is assumed that the model is trained. The output would be the translation
         of the input

         :param src: a PyTorch tensor of shape (N,T)

         :returns: the model outputs. Should be scores of shape (N,T,output_size).
         """
        #############################################################################
        # TODO:
        # Deliverable 5: You will be calling the transformer forward function to    #
        # generate the translation for the input.                                   #
        #############################################################################
        with torch.no_grad():
            outputs = torch.zeros(src.size(0), self.max_length, self.output_size, device=src.device)  # Initialize outputs tensor
            tgt = torch.full((src.size(0), self.max_length), self.pad_idx, dtype=torch.long, device=src.device)
            tgt[:, 0] = 2  # <sos> token
            # initially set outputs as a tensor of zeros with dimensions (batch_size, seq_len, output_size)
            # initially set tgt as a tensor of <pad> tokens with dimensions (batch_size, seq_len)
            for t in range(self.max_length):
                # Generate output for the current time step
                output = self.forward(src, tgt)  # Get the output scores for the current target sequence
                outputs[:, t, :] = output[:, t, :]  # Store the output scores for the current time step
                if t+1 < self.max_length:
                    predicted_token = output[:, t, :].argmax(dim=-1)  # Get the predicted tokens by taking the argmax over the output scores
                    tgt[:, t+1] = predicted_token  # Update the target sequence with the predicted tokens for the next iteration


        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs

    def add_start_token(self, batch_sequences, start_token=2):
        """
            add start_token to the beginning of batch_sequence and shift other tokens to the right
            if batch_sequences starts with two consequtive <sos> tokens, return the original batch_sequence

            example1:
            batch_sequence = [[<sos>, 5,6,7]]
            returns:
                [[<sos>,<sos>, 5,6]]

            example2:
            batch_sequence = [[<sos>, <sos>, 5,6,7]]
            returns:
                [[<sos>, <sos>, 5,6,7]]
        """
        def has_consecutive_start_tokens(tensor, start_token):
            """
                return True if the tensor has two consecutive start tokens
            """
            consecutive_start_tokens = torch.tensor([start_token, start_token], dtype=tensor.dtype,
                                                    device=tensor.device)

            # Check if the first two tokens in each sequence are equal to consecutive start tokens
            is_consecutive_start_tokens = torch.all(tensor[:, :2] == consecutive_start_tokens, dim=1)

            # Return True if all sequences have two consecutive start tokens at the beginning
            return torch.all(is_consecutive_start_tokens).item()

        if has_consecutive_start_tokens(batch_sequences, start_token):
            return batch_sequences

        # Clone the input tensor to avoid modifying the original data
        modified_sequences = batch_sequences.clone()

        # Create a tensor with the start token and reshape it to match the shape of the input tensor
        start_token_tensor = torch.tensor(start_token, dtype=modified_sequences.dtype, device=modified_sequences.device)
        start_token_tensor = start_token_tensor.view(1, -1)

        # Shift the words to the right
        modified_sequences[:, 1:] = batch_sequences[:, :-1]

        # Add the start token to the first word in each sequence
        modified_sequences[:, 0] = start_token_tensor

        return modified_sequences

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True