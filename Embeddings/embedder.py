import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List

class Embedder:
    """
        loading embedding model
    """

    def __init__(
            self,
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            device: str | None = None,
    ):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed(self, texts: List[str]) -> torch.Tensor:
        """
        returns:
            Tensor of shape [len(texts), hidden_dim]
        """
        inputs = self.tokenizer(
            texts,
            padding= True,
            truncation=True,
            return_tensors='pt',
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        token_embeddings= outputs.last_hidden_state  

        attention_mask= inputs['attention_mask'].unsqueeze(-1).float()  
        masked_token_embeddings= token_embeddings * attention_mask  
        sum_embeddings= torch.sum(masked_token_embeddings, dim=1)  
        count= torch.clamp(attention_mask.sum(dim=1), min=1e-9)  
        embeddings= sum_embeddings / count  
        
        normalized_embeddings= F.normalize(embeddings, p=2, dim=1)

        return normalized_embeddings.cpu()