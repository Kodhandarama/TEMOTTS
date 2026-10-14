"""Emotion language model used in Stage II (pruning + emotion text embeddings).

Paper Sec. 3.2: a fine-tuned DistilRoBERTa-base emotion classifier
(michellejieli/emotion_text_classifier). Class probabilities drive emotion data
pruning; mean pooling over the last layer gives the emotion text embedding.
"""
import torch

LM_NAME = "michellejieli/emotion_text_classifier"


class EmotionTextEncoder:
    def __init__(self, device="cpu", name=LM_NAME):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval().to(device)
        self.device = device
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def __call__(self, texts, batch_size=32):
        """Returns (embeddings [N, 768], class probabilities [N, n_classes])."""
        embs, probs = [], []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i : i + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            out = self.model(**enc, output_hidden_states=True)
            last = out.hidden_states[-1]  # (B, T, 768) last transformer layer
            mask = enc["attention_mask"].unsqueeze(-1).to(last.dtype)
            embs.append(((last * mask).sum(1) / mask.sum(1)).cpu())  # mean pooling over real tokens
            probs.append(out.logits.softmax(-1).cpu())
        return torch.cat(embs), torch.cat(probs)
