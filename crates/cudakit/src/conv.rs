//! Library-independent convolution geometry

use crate::Math;

/// An NCHW 2-D convolution on FP32 buffers (cross-correlation, as in PyTorch `Conv2d`)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv2d {
    /// Batch size `n`
    pub batch: usize,
    /// Input channels `c`
    pub in_channels: usize,
    /// Output channels `k`
    pub out_channels: usize,
    /// Input height and width
    pub input: [usize; 2],
    /// Filter height and width
    pub kernel: [usize; 2],
    /// Zero padding on each side of height and width
    pub padding: [usize; 2],
    /// Stride along height and width
    pub stride: [usize; 2],
    /// Dilation along height and width
    pub dilation: [usize; 2],
    /// Precision of the convolution; FP32 unless the caller opts into TF32
    pub math: Math,
}

impl Conv2d {
    /// Output height and width, using the PyTorch formula
    pub fn output(&self) -> [usize; 2] {
        [0, 1].map(|axis| {
            let span = self.dilation[axis] * (self.kernel[axis].saturating_sub(1)) + 1;
            (self.input[axis] + 2 * self.padding[axis]).saturating_sub(span)
                / self.stride[axis].max(1)
                + 1
        })
    }

    /// Input shape `[n, c, h, w]`
    pub fn input_shape(&self) -> [usize; 4] {
        [self.batch, self.in_channels, self.input[0], self.input[1]]
    }

    /// Filter shape `[k, c, r, s]`
    pub fn filter_shape(&self) -> [usize; 4] {
        [
            self.out_channels,
            self.in_channels,
            self.kernel[0],
            self.kernel[1],
        ]
    }

    /// Output shape `[n, k, p, q]`
    pub fn output_shape(&self) -> [usize; 4] {
        let [p, q] = self.output();
        [self.batch, self.out_channels, p, q]
    }
}
