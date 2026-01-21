"""Provide ``get_heights`` function."""

from collections.abc import Callable

from localuf.type_aliases import DecodingScheme


def get_heights(
        d: int,
        global_batch_slenderness: int,
        scheme: DecodingScheme,
        get_commit_height: Callable[[int], int] | None,
        get_buffer_height: Callable[[int], int] | None,
):
    """Get height inputs for ``Code.__init__`` based on decoding scheme and multipliers.
    
    
    :param d: code distance.
    :param global_batch_slenderness: the slenderness in case ``scheme == 'global batch'``.
    :param scheme: decoding scheme.
    :param get_commit_height: a function with input ``d`` that outputs commit height.
    :param get_buffer_height: a function with input ``d`` that outputs buffer height.
    """
    window_height = d*global_batch_slenderness if scheme == 'global batch' else None

    if get_commit_height is None:
        if scheme == 'forward':
            commit_height = d
        elif scheme == 'frugal':
            commit_height = 1
        else:  # 'batch' in scheme
            commit_height = None
    else:
        commit_height = get_commit_height(d)

    if get_buffer_height is None:
        if scheme == 'forward':
            buffer_height = d
        elif scheme == 'frugal':
            buffer_height = 2*(d//2)
        else:  # 'batch' in scheme
            buffer_height = None
    else:
        buffer_height = get_buffer_height(d)

    return window_height, commit_height, buffer_height