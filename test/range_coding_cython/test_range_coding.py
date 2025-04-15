import numpy as np
import pytest

from src.range_coding_cython import encode_nxk, decode_nxk


def calculate_sample_cdf(alphabet_size: int, rng: int, precision: int = 16):
    freqs = rng.uniform(size=alphabet_size)
    pdf = freqs / freqs.sum(axis=-1, keepdims=True)
    cdf_base = 2**precision
    pdf = (pdf * (cdf_base - alphabet_size)) + 1
    cdf = np.append(np.zeros((*pdf.shape[:-1], 1)), pdf.cumsum(axis=-1), axis=-1)
    return cdf.astype(np.uint32)


@pytest.mark.parametrize("alphabet_size", [4, 16, 64])
@pytest.mark.parametrize("length", [16, 1024, 16384])
@pytest.mark.parametrize("seed", [0, 42, 123456789])
def test_range_coding(alphabet_size: int, length: int, seed: int):
    rng = np.random.default_rng(seed)
    syms = rng.integers(low=0, high=alphabet_size, size=length, dtype=np.uint32)

    num_cdfs = min(64, length // 4)
    cdfs = np.array([calculate_sample_cdf(alphabet_size, rng) for _ in range(num_cdfs)])
    cdf_indices = np.arange(length, dtype=np.uint32) % num_cdfs

    data = bytearray()
    data = encode_nxk(syms, cdfs, cdf_indices, data)
    buff = np.zeros_like(syms)
    out = decode_nxk(buff, cdfs, cdf_indices, data)
    assert (out == syms).all(), (
        f"Test failed for alphabet_size={alphabet_size}, length={length}, seed={seed}."
    )
