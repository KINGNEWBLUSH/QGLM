from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import cos_sim


class SentenceBert():
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def encode(self, sentences):
        return self.model.encode(sentences, convert_to_tensor=True)

    def cos_sim(self, embeddings1, embeddings2):
        return cos_sim(embeddings1, embeddings2)
    
    def pair_sim(self, sentecne1, sentence2):
        embedding1 = self.encode(sentecne1)
        embedding2 = self.encode(sentence2)
        return self.cos_sim(embedding1, embedding2)

    def set_sim(self, sentence1: str, sentence_set: list):
        if (type(sentence_set) is not list):
            sentence_set = [sentence_set]
        embeddings1 = self.encode([sentence1])
        embeddings2 = self.encode(sentence_set)
        return self.cos_sim(embeddings1, embeddings2)

if __name__ == "__main__":
    sbert = SentenceBert()
    s1 = """一款手机的原价是8000元。商家的促销活动分为两步:1.先把价格降为原价的75%，再对结果打八折。这款手机的最终售价是多少?"""
    s2 = """一件上衣原价128元，现在打8.8销售，这件上衣便宜了多少钱?"""
    print(sbert.pair_sim(s1, s2))