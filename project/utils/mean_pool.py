import torch

def mean_pooling_reference_encoder(model_output, attention_mask):
    """
    Processing of the reference encoder output.
    Input:
        model_output[0]: token embeddings, shape (batch_size, sequence_length, hidden_dim)
        attention_mask: shape (batch_size, sequence_length)
    Output:
        embeddings of sequences, shape (batch_size, hidden_dim)
    """
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings; shape (batch_size, sequence_length, hidden_dim)
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
