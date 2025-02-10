# from src.models.utils.hooks import GPSHook
# from src.models.gps import GPS
# from src.models.model import train

# class AttentionCorrelation:
#     """
#     Takes in models, initializes with different hyperparameter sets minus number of layers which must stay constant
#     Takes in data

#     For each trained model, store attention weights, and then compute pearson correlation between them
#     """

#     # __init__(self, data, num_classes, hidden_channels=4, pe_channels=4, num_attention_heads=1, num_layers=1, observe_attention=False, return_logits=True)
#     def __init__(self, num_gps_models, data, num_classes, num_layers=2):
#         self.data = data
#         self.weights = {}
