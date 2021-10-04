from functools import lru_cache
import numpy as np

class ZgyLoader:
    def __init__(self, filehandle):
        self.filehandle = filehandle
        self.n_ilines, self.n_xlines, self.n_samples = self.filehandle.size

    @lru_cache(maxsize=1)
    def load_inline_chunk(self, il_idx):
        assert il_idx % 64 == 0
        buf = np.zeros((64, self.n_xlines, self.n_samples), dtype=np.float32)
        self.filehandle.read((il_idx, 0, 0), buf, zeroed_data=True)
        return buf

    @lru_cache(maxsize=1)
    def load_crossline_chunk(self, xl_idx):
        assert xl_idx % 64 == 0
        buf = np.zeros((self.n_ilines, 64, self.n_samples), dtype=np.float32)
        self.filehandle.read((0, xl_idx, 0), buf, zeroed_data=True)
        return buf

    @lru_cache(maxsize=1)
    def load_zslice_chunk(self, z_idx):
        assert z_idx % 64 == 0
        buf = np.zeros((self.n_ilines, self.n_xlines, 64), dtype=np.float32)
        self.filehandle.read((0, 0, z_idx), buf, zeroed_data=True)
        return buf

    @lru_cache(maxsize=1)
    def load_trace_chunk(self, il, xl):
        assert il % 64 == 0 and xl % 64 == 0
        buf = np.zeros((64, 64, self.n_samples), dtype=np.float32)
        self.filehandle.read((il, xl, 0), buf, zeroed_data=True)
        return buf
