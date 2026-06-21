use std::ops::{AddAssign, Index, IndexMut};

use ndarray::{
    Array1, Array2, Array4, ArrayBase, ArrayD, ArrayViewD, ArrayViewMutD, Axis, Dim, OwnedRepr,
    Slice, s,
};
use rand::rng;
use rand_distr::{Distribution, Normal};

pub struct Conv2DNonBatch {
    kernel: Array4<f32>,
    kernel_grad: Array4<f32>,
    bias: Array1<f32>,
    bias_grad: Array1<f32>,
    in_channel: usize,
    out_channel: usize,
    kernel_size: usize,

    e: f32,
    t: usize,
    v_kernel: Array4<f32>,
    m_kernel: Array4<f32>,
    v_bias: Array1<f32>,
    m_bias: Array1<f32>,
    b_1: f32,
    b_2: f32,
}

impl Conv2DNonBatch {
    pub fn adam_optim(&mut self, lr: f32) {
        // weight
        let m = &self.m_kernel * self.b_1 + (1. - self.b_1) * &self.kernel_grad;
        let correction_m = &m / (1. - self.b_1.powi(self.t as i32));

        let v = &self.v_kernel * self.b_2 + (1. - self.b_2) * self.kernel_grad.pow2();
        let correction_v = &v / (1. - self.b_2.powi(self.t as i32));

        self.kernel = &self.kernel.view() - lr / (self.e + correction_v.sqrt()) * correction_m;
        self.m_kernel = m;
        self.v_kernel = v;

        // bias
        let m = &self.m_bias * self.b_1 + (1. - self.b_1) * &self.bias_grad;
        let correction_m = &m / (1. - self.b_1.powi(self.t as i32));

        let v = &self.v_bias * self.b_2 + (1. - self.b_2) * self.bias_grad.pow2();
        let correction_v = &v / (1. - self.b_2.powi(self.t as i32));

        self.bias = &self.bias.view() - lr / (self.e + correction_v.sqrt()) * correction_m;
        self.m_bias = m;
        self.v_bias = v;

        self.t += 1;
    }

    pub fn zero_grad(&mut self) {
        self.kernel_grad = Array4::zeros([
            self.out_channel,
            self.in_channel,
            self.kernel_size,
            self.kernel_size,
        ]);

        self.bias_grad = Array1::<f32>::zeros([self.out_channel]);
    }

    pub fn new(in_channel: usize, out_channel: usize, kernel_size: usize) -> Conv2DNonBatch {
        let len = out_channel * in_channel * kernel_size * kernel_size;

        // He
        let std = (2. / (in_channel * kernel_size * kernel_size) as f32).sqrt();
        let normal = Normal::new(0., std).unwrap();
        let mut rng = rng();

        Self {
            kernel: Array4::<f32>::from_shape_vec(
                [out_channel, in_channel, kernel_size, kernel_size],
                (0..len).map(|_| normal.sample(&mut rng)).collect(),
            )
            .unwrap(),
            kernel_grad: Array4::<f32>::zeros([out_channel, in_channel, kernel_size, kernel_size]),
            bias: Array1::<f32>::zeros([out_channel]),
            bias_grad: Array1::<f32>::zeros([out_channel]),
            in_channel,
            kernel_size,
            out_channel,

            e: 0.00000001,
            t: 1,
            v_kernel: Array4::<f32>::zeros([out_channel, in_channel, kernel_size, kernel_size]),
            m_kernel: Array4::<f32>::zeros([out_channel, in_channel, kernel_size, kernel_size]),
            v_bias: Array1::<f32>::zeros([out_channel]),
            m_bias: Array1::<f32>::zeros([out_channel]),
            b_1: 0.9,
            b_2: 0.999,
        }
    }

    pub fn backward(
        &mut self,
        input: ArrayViewD<f32>,
        mut input_grad: Option<ArrayViewMutD<f32>>,
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
                        self.kernel_grad
                            .slice_mut(s![
                                out_kernel_idx..out_kernel_idx + 1,
                                channel_idx..channel_idx + 1,
                                ..,
                                ..
                            ])
                            .add_assign(&d_kernel);

                        if let Some(input_grad) = &mut input_grad {
                            let d_input_kernel = kernel_matrix * *grad;
                            input_grad
                                .slice_mut(s![
                                    channel_idx..channel_idx + 1,
                                    row..row + self.kernel_size,
                                    coll..coll + self.kernel_size
                                ])
                                .add_assign(&d_input_kernel);
                        }
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

        // let out = (0..self.out_channel)
        //     .into_par_iter()
        //     .map(|out_kernel_idx| {
        //         let mut kernel_save = vec![];
        //         for row in 0..matrix_size - self.kernel_size + 1 {
        //             for coll in 0..matrix_size - self.kernel_size + 1 {
        //                 //
        //                 let mut acc = Array2::<f32>::zeros([self.kernel_size, self.kernel_size]);
        //                 for channel_idx in 0..channel_size {
        //                     let kernel = self.kernel.slice(s![
        //                         out_kernel_idx..out_kernel_idx + 1,
        //                         channel_idx..channel_idx + 1,
        //                         ..,
        //                         ..
        //                     ]);
        //                     let kernel_matrix = kernel
        //                         .to_shape([self.kernel_size, self.kernel_size])
        //                         .unwrap();

        //                     let slice = input.slice(s![
        //                         channel_idx..channel_idx + 1,
        //                         row..row + self.kernel_size,
        //                         coll..coll + self.kernel_size
        //                     ]);
        //                     let slice_matrix = slice
        //                         .to_shape([self.kernel_size, self.kernel_size])
        //                         .unwrap();

        //                     let result = slice_matrix * kernel_matrix;
        //                     acc.add_assign(&result);
        //                 }
        //                 let sum = acc.sum() + self.bias.index([out_kernel_idx]);
        //                 kernel_save.push(sum);
        //             }
        //         }
        //         kernel_save
        //     })
        //     .collect::<Vec<Vec<f32>>>()
        //     .concat();

        let mut out = vec![];
        for out_kernel_idx in 0..self.out_channel {
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
