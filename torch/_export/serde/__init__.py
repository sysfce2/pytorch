import contextlib
from collections.abc import Generator


@contextlib.contextmanager
def unsafe_export_save_load() -> Generator[None]:
    """Context manager that enables serialization and deserialization of
    arbitrary Python callables in exported programs.

    By default, :func:`torch.export.save` and :func:`torch.export.load` only
    support serializing standard PyTorch operators (``torch.ops.*``) and
    higher-order operators. Exported programs that contain plain Python callable
    nodes -- such as predispatch wrapper functions from ``torch._functorch``
    (e.g., ``_jvp_increment_nesting``, ``_vmap_increment_nesting``) -- will raise
    :class:`SerializeError` during save or load.

    This context manager opts into serialization of these callable nodes. It is
    intended for advanced use cases (e.g., models using ``torch.func.jvp`` or
    ``torch.vmap``) where the caller accepts the following tradeoff:

    .. warning::
        **No backwards compatibility guarantee.** Serialized artifacts produced
        under this context manager may not be loadable across different PyTorch
        versions, because the callable targets are resolved by module path
        (e.g., ``torch._functorch.predispatch._jvp_increment_nesting``). If
        PyTorch renames, moves, or removes these internal functions, loading will
        fail.

    Args:
        None

    Example::

        import torch
        from torch.export import export, save, load, unsafe_export_save_load


        class JVPModel(torch.nn.Module):
            def forward(self, x, v):
                return torch.func.jvp(lambda x: x.sin(), (x,), (v,))


        ep = export(JVPModel(), (torch.randn(3), torch.randn(3)), strict=False)

        # Without the context manager, this raises SerializeError
        # because the graph contains predispatch callable nodes.
        with unsafe_export_save_load():
            save(ep, "model.pt2")
            loaded_ep = load("model.pt2")
    """
    import torch._export.config as _export_config

    with _export_config.patch(allow_unsafe_callable_serialization=True):
        yield
