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

    // println!("out");
    // println!("{}\n", out);
    // println!("out grad");
    // println!("{}\n", out_grad);

    conv2d.backpropagation(input.view(), &mut input_grad, out_grad.view());
    // println!("input grad");
    // println!("{}", input_grad);
}
