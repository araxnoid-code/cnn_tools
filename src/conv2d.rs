use std::ops::{AddAssign, Index};

use ndarray::{Array2, ArrayBase, ArrayD, ArrayViewD, Axis, Dim, OwnedRepr, Slice, s};

pub struct Conv2DNonBatch {
    kernels: Vec<Vec<ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>, f32>>>,
    grads: Vec<Vec<ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>, f32>>>,
    in_channel: usize,
    out_channel: usize,
    kernel_size: usize,
}

impl Conv2DNonBatch {
    pub fn init(in_channel: usize, out_channel: usize, kernel_size: usize) -> Conv2DNonBatch {
        let mut kernels = vec![];
        let mut grads = vec![];
        let mut count = 0.;
        for _ in 0..out_channel {
            let mut out_kernel = vec![];
            let mut out_grad = vec![];
            for _ in 0..in_channel {
                let kernel = Array2::<f32>::from_shape_vec(
                    [kernel_size, kernel_size],
                    (0..kernel_size * kernel_size)
                        .map(|_| {
                            count += 1.;
                            count as f32 * 0.1
                        })
                        .collect::<Vec<f32>>(),
                )
                .unwrap();
                out_kernel.push(kernel);

                let grad = Array2::<f32>::zeros([kernel_size, kernel_size]);
                out_grad.push(grad);
            }
            kernels.push(out_kernel);
            grads.push(out_grad);
        }

        Self {
            in_channel,
            kernels: kernels,
            kernel_size,
            grads: grads,
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

        for (out_idx, out_kernel) in self.kernels.iter().enumerate() {
            //
            for row in 0..matrix_size - self.kernel_size + 1 {
                for coll in 0..matrix_size - self.kernel_size + 1 {
                    //
                    for channel_idx in 0..channel_size {
                        let kernel = &out_kernel[channel_idx];
                        let matrix = input.slice_axis(
                            Axis(0),
                            Slice::new(channel_idx as isize, Some(channel_idx as isize + 1), 1),
                        );
                        let mut matrix_grad = input_grad.slice_axis_mut(
                            Axis(0),
                            Slice::new(channel_idx as isize, Some(channel_idx as isize + 1), 1),
                        );

                        let matrix = matrix.index_axis(Axis(0), 0);
                        let mut matrix_grad = matrix_grad.index_axis_mut(Axis(0), 0);
                        let slice = matrix.slice(s![
                            row..row + self.kernel_size,
                            coll..coll + self.kernel_size
                        ]);
                        let mut grad_slice = matrix_grad.slice_mut(s![
                            row..row + self.kernel_size,
                            coll..coll + self.kernel_size
                        ]);

                        let grad = gradient.index([out_idx, row, coll]);
                        self.grads[out_idx][channel_idx].add_assign(&(slice.to_owned() * *grad));
                        grad_slice.add_assign(&(kernel * *grad));
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
        for out_kernel in &self.kernels {
            //
            for row in 0..matrix_size - self.kernel_size + 1 {
                for coll in 0..matrix_size - self.kernel_size + 1 {
                    //
                    let mut acc: Option<ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>, f32>> = None;
                    for channel_idx in 0..channel_size {
                        let kernel = &out_kernel[channel_idx].view();

                        let slice = input.slice(s![
                            channel_idx..channel_idx + 1,
                            row..row + self.kernel_size,
                            coll..coll + self.kernel_size
                        ]);

                        let result = &slice * kernel;
                        if let Some(acc) = &mut acc {
                            acc.add_assign(&result);
                        } else {
                            acc = Some(result)
                        }
                    }
                    let sum = acc.unwrap().sum();
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
