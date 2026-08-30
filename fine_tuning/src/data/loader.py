from typing import Any

from torch.utils.data import default_collate


def fomo26_collate(batch: list) -> tuple:
    """Universal collate function for ``FOMO26Dataset`` batches.

    Handles both ``(image, label)`` and ``(image, label, meta)`` sample
    formats. ``image`` and ``label`` are always stacked with
    ``default_collate`` since they have a fixed shape after transforms.

    No stacking of ``meta``. Simply returned as a list of dictionaries, one
    per sample.

    Parameters
    ----------
    batch:
        List of samples, each either ``(image, label)`` or
        ``(image, label, meta)``, as produced by ``FOMO26Dataset.__getitem__``.

    Returns
    -------
    tuple
        ``(images, labels)`` or ``(images, labels, meta)``, matching the
        input sample structure.
    """
    has_meta = len(batch[0]) == 3

    images = default_collate([sample[0] for sample in batch])
    labels = default_collate([sample[1] for sample in batch])

    if not has_meta:
        return images, labels

    metas = [sample[2] for sample in batch]
    return images, labels, metas