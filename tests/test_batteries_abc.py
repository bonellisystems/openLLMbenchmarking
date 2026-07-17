import pytest
from llmtest import batteries


def test_registry_rejects_unknown():
    with pytest.raises(KeyError):
        batteries.get(99)


def test_register_and_get_roundtrip():
    @batteries.register
    class Dummy(batteries.Battery):
        id = 98
        def plan(self, cfg, store, model_filter=None):
            return []
        def execute(self, item, ctx):
            return []
    assert isinstance(batteries.get(98), Dummy)
