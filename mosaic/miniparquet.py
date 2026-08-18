from __future__ import annotations

import ctypes
import math
from pathlib import Path
from typing import Any

import numpy as np
from thrift.transport import TTransport
from thrift.protocol import TCompactProtocol
from thrift.Thrift import TType

PARQUET_TYPES = {0: np.bool_, 1: np.int32, 2: np.int64, 4: np.float32, 5: np.float64}


def _read_thrift_value(proto: TCompactProtocol.TCompactProtocol, ttype: int) -> Any:
    if ttype == TType.BOOL: return proto.readBool()
    if ttype == TType.BYTE: return proto.readByte()
    if ttype == TType.I16: return proto.readI16()
    if ttype == TType.I32: return proto.readI32()
    if ttype == TType.I64: return proto.readI64()
    if ttype == TType.DOUBLE: return proto.readDouble()
    if ttype == TType.STRING: return proto.readBinary()
    if ttype == TType.STRUCT:
        out: dict[int, Any] = {}; proto.readStructBegin()
        while True:
            _, field_type, field_id = proto.readFieldBegin()
            if field_type == TType.STOP: break
            out[field_id] = _read_thrift_value(proto, field_type); proto.readFieldEnd()
        proto.readStructEnd(); return out
    if ttype == TType.LIST:
        elem_type, size = proto.readListBegin(); out = [_read_thrift_value(proto, elem_type) for _ in range(size)]; proto.readListEnd(); return out
    if ttype == TType.SET:
        elem_type, size = proto.readSetBegin(); out = [_read_thrift_value(proto, elem_type) for _ in range(size)]; proto.readSetEnd(); return out
    if ttype == TType.MAP:
        key_type, value_type, size = proto.readMapBegin(); out = [(_read_thrift_value(proto, key_type), _read_thrift_value(proto, value_type)) for _ in range(size)]; proto.readMapEnd(); return out
    raise NotImplementedError(f"Unsupported Thrift type {ttype}")


def _read_compact_struct(payload: bytes) -> tuple[dict[int, Any], int]:
    transport = TTransport.TMemoryBuffer(payload)
    protocol = TCompactProtocol.TCompactProtocol(transport)
    value = _read_thrift_value(protocol, TType.STRUCT)
    return value, transport._buffer.tell()


class _Snappy:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libsnappy.so.1")
        self.lib.snappy_uncompressed_length.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.snappy_uncompress.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)]

    def decompress(self, data: bytes) -> bytes:
        output_length = ctypes.c_size_t()
        result = self.lib.snappy_uncompressed_length(data, len(data), ctypes.byref(output_length))
        if result != 0: raise RuntimeError(f"snappy_uncompressed_length failed with code {result}")
        output = ctypes.create_string_buffer(output_length.value)
        written = ctypes.c_size_t(output_length.value)
        result = self.lib.snappy_uncompress(data, len(data), output, ctypes.byref(written))
        if result != 0: raise RuntimeError(f"snappy_uncompress failed with code {result}")
        return output.raw[: written.value]

_SNAPPY = _Snappy()


def _decode_uvarint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0; shift = 0
    while True:
        byte = data[offset]; offset += 1; value |= (byte & 0x7F) << shift
        if byte < 0x80: return value, offset
        shift += 7
        if shift > 63: raise ValueError("Invalid varint")


def _unpack_bitpacked(data: bytes, bit_width: int, count: int) -> np.ndarray:
    if bit_width == 0: return np.zeros(count, dtype=np.int64)
    out = np.empty(count, dtype=np.int64); bit_pos = 0; mask = (1 << bit_width) - 1
    for i in range(count):
        byte_index = bit_pos >> 3; shift = bit_pos & 7; nbytes = (shift + bit_width + 7) >> 3
        word = int.from_bytes(data[byte_index:byte_index+nbytes], "little")
        out[i] = (word >> shift) & mask; bit_pos += bit_width
    return out


def _decode_rle_bitpacked_hybrid(data: bytes, bit_width: int, count: int, *, has_bit_width_prefix: bool=False) -> np.ndarray:
    offset = 0
    if has_bit_width_prefix: bit_width = data[0]; offset = 1
    values: list[np.ndarray] = []; produced = 0; value_bytes = max(1, math.ceil(bit_width / 8))
    while produced < count:
        header, offset = _decode_uvarint(data, offset)
        if header & 1 == 0:
            run_length = header >> 1; raw = data[offset:offset+value_bytes]; offset += value_bytes
            value = int.from_bytes(raw, "little") if bit_width else 0; take = min(run_length, count-produced)
            values.append(np.full(take, value, dtype=np.int64)); produced += take
        else:
            groups = header >> 1; run_length = groups * 8; nbytes = groups * bit_width
            decoded = _unpack_bitpacked(data[offset:offset+nbytes], bit_width, run_length); offset += nbytes
            take = min(run_length, count-produced); values.append(decoded[:take]); produced += take
    return np.concatenate(values) if values else np.empty(0, dtype=np.int64)


def _decode_plain(payload: bytes, parquet_type: int, count: int) -> np.ndarray:
    if parquet_type == 0: return _unpack_bitpacked(payload, 1, count).astype(bool)
    dtype = PARQUET_TYPES.get(parquet_type)
    if dtype is None: raise NotImplementedError(f"PLAIN decoding for Parquet type {parquet_type}")
    return np.frombuffer(payload, dtype=np.dtype(dtype).newbyteorder("<"), count=count).copy()


def _max_definition_level(schema_element: dict[int, Any]) -> int:
    repetition_type = schema_element.get(3, 0)
    return 0 if repetition_type == 0 else 1


def _decode_levels(data: bytes, count: int, max_level: int) -> tuple[np.ndarray, int]:
    if max_level == 0: return np.zeros(count, dtype=np.int8), 0
    encoded_length = int.from_bytes(data[:4], "little")
    bit_width = max(1, math.ceil(math.log2(max_level + 1)))
    levels = _decode_rle_bitpacked_hybrid(data[4:4+encoded_length], bit_width, count)
    return levels.astype(np.int8), 4 + encoded_length


class ParquetFile:
    """Narrow Parquet reader for the flat numeric Job-SDF files.

    Supports multiple row groups, Snappy, PLAIN dictionaries and RLE_DICTIONARY data pages.
    """
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path); self.data = self.path.read_bytes()
        if self.data[:4] != b"PAR1" or self.data[-4:] != b"PAR1": raise ValueError(f"Not a Parquet file: {self.path}")
        footer_length = int.from_bytes(self.data[-8:-4], "little"); footer = self.data[-8-footer_length:-8]
        self.metadata, _ = _read_compact_struct(footer); self.num_rows = int(self.metadata[3]); self.schema = self.metadata[2]; self.row_groups = self.metadata[4]
        self.column_names = [element[4].decode("utf-8") for element in self.schema[1:]]
        self.schema_by_name = {element[4].decode("utf-8"): element for element in self.schema[1:]}
        self.chunks_by_name = {name: [] for name in self.column_names}
        for row_group in self.row_groups:
            for chunk in row_group[1]:
                name = chunk[3][3][0].decode("utf-8")
                self.chunks_by_name[name].append(chunk)

    def _decompress(self, codec: int, data: bytes, expected_size: int) -> bytes:
        if codec == 0: return data
        if codec == 1:
            out = _SNAPPY.decompress(data)
            if len(out) != expected_size: raise ValueError("Snappy output length mismatch")
            return out
        raise NotImplementedError(f"Compression codec {codec} is not supported")

    def _read_chunk(self, name: str, chunk: dict[int, Any]) -> np.ndarray:
        meta = chunk[3]; parquet_type = int(meta[1]); codec = int(meta[4]); num_values = int(meta[5]); data_offset = int(meta[9]); dictionary_offset = meta.get(11)
        start = int(dictionary_offset if dictionary_offset is not None else data_offset); end = start + int(meta[7]); pos = start
        dictionary: np.ndarray | None = None; output: list[np.ndarray] = []; schema_element = self.schema_by_name[name]; max_def = _max_definition_level(schema_element); decoded_count = 0
        while pos < end and decoded_count < num_values:
            header, header_size = _read_compact_struct(self.data[pos:min(len(self.data), pos+65536)]); pos += header_size
            page_type = int(header[1]); uncompressed_size = int(header[2]); compressed_size = int(header[3]); payload = self.data[pos:pos+compressed_size]; pos += compressed_size
            raw = self._decompress(codec, payload, uncompressed_size)
            if page_type == 2:
                dictionary_header = header[7]; dictionary_count = int(dictionary_header[1]); dictionary_encoding = int(dictionary_header[2])
                if dictionary_encoding != 0: raise NotImplementedError("Only PLAIN dictionary pages are supported")
                dictionary = _decode_plain(raw, parquet_type, dictionary_count); continue
            if page_type == 0:
                data_header = header[5]; page_values = int(data_header[1]); encoding = int(data_header[2]); _, level_bytes = _decode_levels(raw, page_values, max_def); value_payload = raw[level_bytes:]
                if encoding == 0: decoded = _decode_plain(value_payload, parquet_type, page_values)
                elif encoding == 8:
                    if dictionary is None: raise ValueError("Dictionary-encoded page without dictionary")
                    indices = _decode_rle_bitpacked_hybrid(value_payload, 0, page_values, has_bit_width_prefix=True); decoded = dictionary[indices]
                else: raise NotImplementedError(f"Data encoding {encoding} is not supported")
                output.append(decoded); decoded_count += len(decoded); continue
            raise NotImplementedError(f"Page type {page_type} is not supported")
        result = np.concatenate(output)[:num_values]
        if len(result) != num_values: raise ValueError(f"Decoded {len(result)} values, expected {num_values}")
        return result

    def read_column(self, name: str) -> np.ndarray:
        pieces = [self._read_chunk(name, chunk) for chunk in self.chunks_by_name[name]]
        result = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        if len(result) != self.num_rows:
            raise ValueError(f"Decoded {len(result)} rows, expected {self.num_rows}")
        return result

    def read(self, columns: list[str] | None=None) -> dict[str, np.ndarray]:
        selected = self.column_names if columns is None else columns
        return {name: self.read_column(name) for name in selected}


def read_parquet(path: str | Path, columns: list[str] | None=None) -> dict[str, np.ndarray]:
    return ParquetFile(path).read(columns)
