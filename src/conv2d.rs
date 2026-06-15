use std::ops::{AddAssign, Index, IndexMut};

use ndarray::{
    Array1, Array2, Array4, ArrayBase, ArrayD, ArrayViewD, Axis, Dim, OwnedRepr, Slice, s,
};

pub struct Conv2DNonBatch {
    kernel: Array4<f32>,
    grads_grad: Array4<f32>,
    bias: Array1<f32>,
    bias_grad: Array1<f32>,
    in_channel: usize,
    out_channel: usize,
    kernel_size: usize,
}

impl Conv2DNonBatch {
    pub fn init(in_channel: usize, out_channel: usize, kernel_size: usize) -> Conv2DNonBatch {
        let len = out_channel * in_channel * kernel_size * kernel_size;

        Self {
            kernel: Array4::<f32>::from_shape_vec(
                [out_channel, in_channel, kernel_size, kernel_size],
                (0..len).map(|i| i as f32).collect(),
            )
            .unwrap(),
            grads_grad: Array4::<f32>::zeros([out_channel, in_channel, kernel_size, kernel_size]),
            bias: Array1::<f32>::ones([out_channel]),
            bias_grad: Array1::<f32>::zeros([out_channel]),
            in_channel,
            kernel_size,
            out_channel,
        }
    }

    pub fn backpropagation(
        &mut self,
        input: ArrayViewD<f32>,
        input_grad: &mut ArrayD<f32>,
        gradient: ArrayViewD<f32>,
    ) {
        let channel_size = self.in_channel;
        let matrix_size = input.shape()[1];

        for out_kernel_idx in 0..self.out_channel {
            for row in 0..matrix_size - self.kernel_size + 1 {
                for coll in 0..matrix_size - self.kernel_size + 1 {
                    let grad = gradient.index([out_kernel_idx, row, coll]);

                    self.bias_grad.index_mut([out_kernel_idx]).add_assign(grad);
                    for channel_idx in 0..channel_size {
                        let kernel = self.kernel.slice(s![
                            out_kernel_idx..out_kernel_idx + 1,
                            channel_idx..channel_idx + 1,
                            ..,
                            ..
                        ]);
                        let kernel_matrix = kernel
                            .to_shape([self.kernel_size, self.kernel_size])
                            .unwrap();

                        let slice = input.slice(s![
                            channel_idx..channel_idx + 1,
                            row..row + self.kernel_size,
                            coll..coll + self.kernel_size
                        ]);
                        let slice_matrix = slice
                            .to_shape([self.kernel_size, self.kernel_size])
                            .unwrap();

                        let d_kernel = slice_matrix * *grad;
                        self.grads_grad
                            .slice_mut(s![
                                out_kernel_idx..out_kernel_idx + 1,
                                channel_idx..channel_idx + 1,
                                ..,
                                ..
                            ])
                            .add_assign(&d_kernel);

                        let d_input_kernel = kernel_matrix * *grad;
                        input_grad
                            .slice_mut(s![
                                channel_idx..channel_idx + 1,
                                row..row + self.kernel_size,
                                coll..coll + self.kernel_size
                            ])
                            .add_assign(&d_input_kernel);
                        // .add_assign((slice * *grad));
                    }
                }
            }
        }
    }

    pub fn forward(
        &self,
        input: ArrayViewD<f32>,
    ) -> ArrayBase<OwnedRepr<f32>, Dim<ndarray::IxDynImpl>, f32> {
        let channel_size = self.in_channel;
        let matrix_size = input.shape()[1];

        let mut out = vec![];
        for out_kernel_idx in 0..self.out_channel {
            //
            for row in 0..matrix_size - self.kernel_size + 1 {
                for coll in 0..matrix_size - self.kernel_size + 1 {
                    //
                    let mut acc = Array2::<f32>::zeros([self.kernel_size, self.kernel_size]);
                    for channel_idx in 0..channel_size {
                        let kernel = self.kernel.slice(s![
                            out_kernel_idx..out_kernel_idx + 1,
                            channel_idx..channel_idx + 1,
                            ..,
                            ..
                        ]);
                        let kernel_matrix = kernel
                            .to_shape([self.kernel_size, self.kernel_size])
                            .unwrap();

                        let slice = input.slice(s![
                            channel_idx..channel_idx + 1,
                            row..row + self.kernel_size,
                            coll..coll + self.kernel_size
                        ]);
                        let slice_matrix = slice
                            .to_shape([self.kernel_size, self.kernel_size])
                            .unwrap();

                        let result = slice_matrix * kernel_matrix;
                        acc.add_assign(&result);
                    }
                    let sum = acc.sum() + self.bias.index([out_kernel_idx]);
                    out.push(sum);
                }
            }
        }

        let output = ArrayD::<f32>::from_shape_vec(
            &[
                self.out_channel,
                matrix_size - self.kernel_size + 1,
                matrix_size - self.kernel_size + 1,
            ][..],
            out,
        )
        .unwrap();

        output
    }
}
