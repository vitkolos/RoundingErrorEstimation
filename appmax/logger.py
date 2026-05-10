import tqdm


def progress(iterable, main=False, **kwargs):
    if not main:
        kwargs['leave'] = False
        kwargs['disable'] = None

    return tqdm.tqdm(iterable, **kwargs)
