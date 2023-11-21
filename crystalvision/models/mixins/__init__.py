try:
    from .compiles import BinaryMixin, CategoricalMixin
except ImportError:
    from crystalvision.models.mixins.compiles import BinaryMixin, CategoricalMixin
