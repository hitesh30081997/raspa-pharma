"""
Compatibility patch for loading nvidia/slu_conformer_transformer_large_slurp with NeMo 2.x.

The SLURP checkpoint's output layer ("classifier") is a plain MultiLayerPerceptron, but NeMo 2.x's
sequence generators expect a TokenClassifier: they read `classifier.mlp.layers` and call
`classifier.with_log_softmax_enabled(...)`. That raises
    AttributeError: 'MultiLayerPerceptron' object has no attribute 'mlp'
This patch gives MultiLayerPerceptron those two members (the MLP acts as its own `.mlp`). It adds no
parameters and changes no outputs, so checkpoints load and save exactly as before.

Call apply() after importing NeMo and before from_pretrained()/restore_from().
Note: NeMo 3.0 no longer ships SLUIntentSlotBPEModel; use nemo_toolkit 2.x (e.g. 2.7.3).
"""
from contextlib import contextmanager


def apply():
    from nemo.collections.common.parts.multi_layer_perceptron import MultiLayerPerceptron as MLP

    if not hasattr(MLP, "mlp"):
        MLP.mlp = property(lambda self: self)
    if not hasattr(MLP, "with_log_softmax_enabled"):
        @contextmanager
        def with_log_softmax_enabled(self, value):
            prev = self.log_softmax
            self.log_softmax = value
            try:
                yield self
            finally:
                self.log_softmax = prev

        MLP.with_log_softmax_enabled = with_log_softmax_enabled
