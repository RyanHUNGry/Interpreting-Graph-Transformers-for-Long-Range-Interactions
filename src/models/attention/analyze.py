import matplotlib.pyplot as plt
from torch import bincount
from collections import Counter
class AttentionAnalyzer:
    def __init__(self, attention_mat):
        self.attention_mat = attention_mat

    def max_node_contribution_distribution(self):
        max_node_contributions = self.attention_mat.argmax(dim=1)
        
        return Counter(max_node_contributions.tolist())