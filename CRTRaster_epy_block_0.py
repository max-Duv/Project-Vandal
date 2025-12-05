import numpy as np
from gnuradio import gr

class blk(gr.sync_block):
    """Raster pattern generator block"""

    def __init__(self):
        gr.sync_block.__init__(
            self,
            name='CRT Raster Generator',
            in_sig=[],
            out_sig=[np.float32]
        )

        # Parameters
        self.samp_rate = 2_500_000
        self.samples_per_line = 4096
        self.sync_len = 128
        self.blank_len = 128
        self.lines_per_frame = 256
        self.pattern = 'lfm'
        self.active_amp = 1.0
        self.sync_level = -0.6
        self.blank_level = 0.05
        self.lfm_f0 = 50_000
        self.lfm_f1 = 500_000
        self.gamma = 1.0

        self.line_idx = 0
        self.sample_idx_in_line = 0
        self.n_active = self.samples_per_line - self.sync_len - self.blank_len
        self.active_lfm, self.active_bars, self.active_grad = self.make_templates()

    def make_templates(self):
        t = np.arange(self.n_active) / self.samp_rate
        k = (self.lfm_f1 - self.lfm_f0) / (self.n_active / self.samp_rate)
        phase = 2 * np.pi * (self.lfm_f0 * t + 0.5 * k * t**2)
        lfm = np.sin(phase)
        bars = np.sign(np.sin(2 * np.pi * 8 * np.arange(self.n_active) / self.n_active))
        grad = np.linspace(0, 1, self.n_active)**self.gamma
        active_lfm = self.active_amp * (0.5 * lfm + 0.5 * grad)
        active_bars = self.active_amp * bars
        active_grad = self.active_amp * grad
        return active_lfm.astype(np.float32), active_bars.astype(np.float32), active_grad.astype(np.float32)

    def get_active(self, line_i):
        if self.pattern == 'lfm':
            return self.active_lfm
        elif self.pattern == 'bars':
            return self.active_bars
        elif self.pattern == 'grad':
            return self.active_grad
        else:
            return self.active_grad

    def work(self, input_items, output_items):
        out = output_items[0]
        noutput_items = len(out)
        filled = 0

        while filled < noutput_items:
            rem_in_line = self.samples_per_line - self.sample_idx_in_line
            n = min(noutput_items - filled, rem_in_line)
            s = self.sample_idx_in_line
            e = s + n

            if e <= self.sync_len:
                seg = np.full(n, self.sync_level, dtype=np.float32)
            elif s < self.sync_len and e > self.sync_len:
                a = self.sync_len - s
                b = n - a
                seg = np.concatenate([
                    np.full(a, self.sync_level, dtype=np.float32),
                    np.full(b, self.blank_level, dtype=np.float32)
                ])
            elif e <= (self.sync_len + self.blank_len):
                seg = np.full(n, self.blank_level, dtype=np.float32)
            elif s < (self.sync_len + self.blank_len) and e > (self.sync_len + self.blank_len):
                a = (self.sync_len + self.blank_len) - s
                b = n - a
                active = self.get_active(self.line_idx)
                seg = np.concatenate([
                    np.full(a, self.blank_level, dtype=np.float32),
                    active[:b]
                ])
            else:
                start_active = s - (self.sync_len + self.blank_len)
                active = self.get_active(self.line_idx)
                seg = active[start_active:start_active + n]

            out[filled:filled + n] = seg
            filled += n
            self.sample_idx_in_line += n

            if self.sample_idx_in_line >= self.samples_per_line:
                self.sample_idx_in_line = 0
                self.line_idx = (self.line_idx + 1) % self.lines_per_frame

        return len(out)
