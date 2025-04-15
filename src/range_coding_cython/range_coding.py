# cython: language_level=3

import cython

from cython import bint, uint, ulonglong as ull, cast


MAX_CODE: uint = 0xFFFFFFFF
ONE_HALF: uint = 0x80000000
ONE_FOURTH: uint = 0x40000000
THREE_FOURTHS: uint = 0xC0000000
PREC_COUNT: uint = 0x10000
PRECISION: ull = 16


@cython.cfunc
def _append_bit(bit: bint, data: bytearray, index: uint) -> uint:
    if index % 8 == 0:
        data.append(0)
    data[index // 8] |= bit << (7 - (index % 8))
    index += 1
    return index


@cython.cfunc
def _append_bit_and_pending(
    bit: bint, data: bytearray, index: uint, pendings: uint
) -> uint:
    index = _append_bit(bit, data, index)

    while pendings > 0:
        index = _append_bit(not bit, data, index)
        pendings -= 1

    return index


@cython.cfunc
def _encode(
    sym: uint,
    cdf: uint[:],
    high: uint,
    low: uint,
    data: bytearray,
    index: uint,
    pendings: uint,
) -> tuple[uint, uint, uint, uint]:
    span: ull = cast(ull, high) - cast(ull, low) + 1
    range_high: ull = cast(ull, cdf[sym + 1])
    range_low: ull = cast(ull, cdf[sym])

    high = low + cast(uint, (span * range_high) >> PRECISION) - 1
    low = low + cast(uint, (span * range_low) >> PRECISION)

    while True:
        if high < ONE_HALF:
            index = _append_bit_and_pending(False, data, index, pendings)
            pendings = 0
        elif low >= ONE_HALF:
            index = _append_bit_and_pending(True, data, index, pendings)
            pendings = 0
        elif low >= ONE_FOURTH and high < THREE_FOURTHS:
            pendings += 1
            high -= ONE_FOURTH
            low -= ONE_FOURTH
        else:
            break

        high = ((high << 1) | 1) & MAX_CODE
        low = (low << 1) & MAX_CODE
    return high, low, index, pendings


@cython.ccall
def encode_nxk(
    syms: uint[:], cdf_table: uint[:, :], cdf_indices: uint[:], data: bytearray
) -> bytes:
    r"""Encode a sequence (n) of symbols with k possible cdfs using range coding.

    Args:
        syms (uint[:]): The symbols to encode.
        cdf_table (uint[:, :]): The CDF table. Each row is a CDF of shape `(alphabet_size + 1,)`. `CDF[0]` is always 0 and `CDF[-1]` is always `2**PRECISION`. CDFs are monotonic and `CDF[i] <= CDF[i + 1] - 1`.
        cdf_indices (uint[:]): The indices of the CDFs for each symbol.
        data (bytearray): The output buffer for the encoded data.

    Returns:
        bytes: The encoded data.
    """

    index: uint = 0
    pendings: uint = 0
    high: uint = MAX_CODE
    low: uint = 0

    for i in range(len(syms)):
        sym: uint = syms[i]
        cdf_index: uint = cdf_indices[i]
        cdf: uint[:] = cdf_table[cdf_index, :]
        high, low, index, pendings = _encode(sym, cdf, high, low, data, index, pendings)

    pendings += 1
    index = _append_bit_and_pending(low >= ONE_FOURTH, data, index, pendings)

    return bytes(data)


@cython.cfunc
def _retrieve_bit(data: bytes, index: uint) -> tuple[bint, uint]:
    if index >= len(data) * 8:
        return False, index
    bit: bint = data[index // 8] & (1 << (7 - (index % 8)))
    index += 1
    return bool(bit), index


@cython.cfunc
def _decode(
    val: uint, cdf: uint[:], high: uint, low: uint, data: bytes, index: uint
) -> tuple[uint, uint, uint, uint, uint]:
    span: ull = cast(ull, high) - cast(ull, low) + 1
    scaled_span = cast(ull, val) - cast(ull, low)
    scaled_val = ((scaled_span + 1) * cast(ull, PREC_COUNT) - 1) // span

    left: uint = 0
    right: uint = cdf.shape[0]
    while right - left > 1:
        mid = (left + right) >> 1
        if cdf[mid] > scaled_val:
            right = mid
        else:
            left = mid
    sym: uint = left

    range_high: ull = cast(ull, cdf[sym + 1])
    range_low: ull = cast(ull, cdf[sym])
    high = low + cast(uint, (span * range_high) >> PRECISION) - 1
    low = low + cast(uint, (span * range_low) >> PRECISION)

    while True:
        if high < ONE_HALF:
            pass
        elif low >= ONE_HALF:
            val -= ONE_HALF
            low -= ONE_HALF
            high -= ONE_HALF
        elif low >= ONE_FOURTH and high < THREE_FOURTHS:
            val -= ONE_FOURTH
            low -= ONE_FOURTH
            high -= ONE_FOURTH
        else:
            break

        high = ((high << 1) | 1) & MAX_CODE
        low = (low << 1) & MAX_CODE
        val = val << 1
        bit, index = _retrieve_bit(data, index)
        val |= bit
    return val, sym, high, low, index


@cython.ccall
def decode_nxk(
    syms: uint[:], cdf_table: uint[:, :], cdf_indices: uint[:], data: bytes
) -> uint[:]:
    r"""Decode a sequence (n) of symbols with k possible cdfs from bytes using range coding.

    Args:
        syms (uint[:]): The output buffer for the decoded symbols.
        cdf_table (uint[:, :]): The CDF table. Each row is a CDF of shape `(alphabet_size + 1,)`. `CDF[0]` is always 0 and `CDF[-1]` is always `2**PRECISION`. CDFs are monotonic and `CDF[i] <= CDF[i + 1] - 1`.
        cdf_indices (uint[:]): The indices of the CDFs for each symbol.
        data (bytes): The input data to decode.

    Returns:
        uint[:]: The decoded symbols.
    """

    index: uint = 0
    high: uint = MAX_CODE
    low: uint = 0
    val: ull = 0

    for _ in range(PRECISION * 2):
        bit, index = _retrieve_bit(data, index)
        val = (val << 1) | bit

    for i in range(len(syms)):
        cdf_index: uint = cdf_indices[i]
        cdf: uint[:] = cdf_table[cdf_index, :]
        val, sym, high, low, index = _decode(val, cdf, high, low, data, index)
        syms[i] = sym
    return syms
