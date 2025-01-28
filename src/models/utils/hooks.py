from torch.nn import MultiheadAttention

class GPSHooks:
    def __init__(self, module):
        attention_weights = list(filter(lambda m: isinstance(m, MultiheadAttention), module.modules()))

        self.hook_removers = []
        self.attention_layer_mapping = {M: i for i, M in enumerate(attention_weights)}
        self.attention_weights = {i: None for i in range(len(attention_weights))}

        for m in module.modules():
            if isinstance(m, MultiheadAttention):
                self.hook_removers.append(m.register_forward_hook(self.attention_hook))

    def remove_hooks(self):
        for hook_remover in self.hook_removers:
            hook_remover.remove()

    # called for each forward pass of MultiheadAttention
    def attention_hook(self, module, _, out):
        attention_layer = self.attention_layer_mapping[module]
        _, w = out
        w = w.detach().clone()
        self.attention_weights[attention_layer] = w
