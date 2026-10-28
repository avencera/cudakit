//! Library-independent convolution geometry

use crate::{Error, Math};

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
    /// Fails unless every size, stride and dilation is positive and the dilated filter
    /// fits the padded input on both axes, which is when [`Self::output`] describes a
    /// real convolution
    pub fn check(&self) -> Result<(), Error> {
        let invalid = |reason: String| Error::Unsupported {
            context: "conv2d geometry",
            reason,
        };
        let sizes = [
            ("batch", self.batch),
            ("in_channels", self.in_channels),
            ("out_channels", self.out_channels),
        ];
        if let Some((name, _)) = sizes.iter().find(|(_, size)| *size == 0) {
            return Err(invalid(format!("{name} is zero")));
        }

        for (axis, name) in [(0, "height"), (1, "width")] {
            let (input, kernel) = (self.input[axis], self.kernel[axis]);
            let (stride, dilation) = (self.stride[axis], self.dilation[axis]);
            if input == 0 || kernel == 0 || stride == 0 || dilation == 0 {
                return Err(invalid(format!(
                    "{name}: input {input}, kernel {kernel}, stride {stride} and dilation {dilation} must all be positive"
                )));
            }

            let span = dilation.saturating_mul(kernel - 1).saturating_add(1);
            let padded = input.saturating_add(self.padding[axis].saturating_mul(2));
            if span > padded {
                return Err(invalid(format!(
                    "{name}: the dilated kernel spans {span}, more than the padded input's {padded}"
                )));
            }
        }

        Ok(())
    }

    /// Output height and width, using the PyTorch formula
    ///
    /// Meaningful only for geometry that passes [`Self::check`]; invalid geometry
    /// saturates rather than panicking
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

#[cfg(test)]
mod tests {
    use super::Conv2d;
    use crate::Math;

    const VALID: Conv2d = Conv2d {
        batch: 1,
        in_channels: 2,
        out_channels: 3,
        input: [5, 4],
        kernel: [3, 3],
        padding: [0, 1],
        stride: [1, 2],
        dilation: [2, 1],
        math: Math::Fp32,
    };

    #[test]
    fn geometry_that_would_saturate_is_rejected() {
        VALID.check().expect("valid geometry");
        assert_eq!(VALID.output(), [1, 2]);

        let too_small = Conv2d {
            input: [2, 4],
            dilation: [1, 1],
            ..VALID
        };
        let zero_stride = Conv2d {
            stride: [1, 0],
            ..VALID
        };
        let no_channels = Conv2d {
            in_channels: 0,
            ..VALID
        };
        for spec in [too_small, zero_stride, no_channels] {
            assert!(spec.check().is_err(), "{spec:?}");
        }
    }
}
