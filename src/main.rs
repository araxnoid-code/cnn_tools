use std::{iter::Sum, ops::Index};

use ndarray::{Array2, ArrayBase, ArrayD, Axis, Dim, IxDynImpl, OwnedRepr, Slice, s};
use neo_cnn::Conv2DNonBatch;

fn main() {
    let input = ArrayD::<f32>::from_shape_vec(
        &[3, 4, 4][..],
        (0..3 * 4 * 4).map(|idx| idx as f32).collect::<Vec<f32>>(),
    )
    .unwrap();
    let mut input_grad = ArrayD::<f32>::zeros(input.shape());

    println!("input");
    println!("{}\n", input);

    let mut conv2d = Conv2DNonBatch::init(3, 3, 2);

    let out = conv2d.forward(input.view());
    let out_grad = ArrayD::<f32>::ones(out.shape());

    println!("out");
    println!("{}\n", out);
    println!("out grad");
    println!("{}\n", out_grad);

    conv2d.backpropagation(input.view(), &mut input_grad, out_grad.view());
    println!("input grad");
    println!("{}", input_grad);
}

fn kernel_initial(
    in_channel: usize,
    out_channel: usize,
    kernel_size: usize,
) -> (
    Vec<Vec<ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>, f32>>>,
    Vec<Vec<ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>, f32>>>,
) {
    let mut kernels = vec![];
    let mut grads = vec![];
    for _ in 0..out_channel {
        let mut out_kernel = vec![];
        let mut out_grad = vec![];
        for _ in 0..in_channel {
            let kernel = Array2::<f32>::from_shape_vec(
                [kernel_size, kernel_size],
                (0..kernel_size * kernel_size)
                    .map(|idx| idx as f32 * 0.001)
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
    (kernels, grads)
}
