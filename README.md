# range_coding_cython
A range coding lib implemented in Cython.


## Install

```bash
pip install range_coding_cython
```

## Usage

```python
import numpy as np
from range_coding_cython import encode_nxk, decode_nxk

# Example usage
syms = np.array([0, 1, 2, 3, 3, 3], dtype=np.uint32)

cdf = [0.0, 0.1, 0.3, 0.6, 1.0]
PRECISION = 16
cdf_table = np.array([cdf]) * (1 << PRECISION) # (1, alphabet_size + 1)
cdf_table = cdf_table.astype(np.uint32)

cdf_indices = np.array([0, 0, 0, 0, 0, 0], dtype=np.uint32)

# Encode
data = bytearray()
data = encode_nxk(syms, cdf_table, cdf_indices, data)

# Decode
buff = np.zeros_like(symbols)
dec_syms = decode_nxk(buff, cdf_table, cdf_indices, data)

# Check if the decoded symbols match the original symbols
assert np.array_equal(dec_syms, syms)
```

## References

- [Range Coding](https://en.wikipedia.org/wiki/Range_coding)
- [Arithmetic Coder](https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html)
- [NeuralCompression](https://github.com/facebookresearch/NeuralCompression)
- [Yet-Another-Entropy-Coding-Library](https://github.com/tongdaxu/YAECL-Yet-Another-Entropy-Coding-Library)

## 