use std::{iter::Sum, ops::Index};

use ndarray::{Array2, ArrayBase, ArrayD, Axis, Dim, IxDynImpl, OwnedRepr, Slice, s};
use neo_cnn::{Conv2DNonBatch, LinaerNonBatch, MaxPooling2DNonBatch, softmax};
use rand::random;

fn main() {
    let input = ArrayD::<f32>::from_shape_vec(&[2, 3][..], vec![1., 4., 2., 7., 5., 9.]).unwrap();
    let mut input_gradient = ArrayD::<f32>::zeros(input.shape());

    println!("{}", input);

    let pred = softmax(input.view(), 1).unwrap();

    // let mut linear = LinaerNonBatch::new(3, 4);

    // let output = linear.forward(input.view());
    // let output_gradinet = ArrayD::<f32>::ones(output.shape());

    // linear.backpropagation(
    //     input.view(),
    //     input_gradient.view_mut(),
    //     output_gradinet.view(),
    // );

    // let input = ArrayD::<f32>::from_shape_vec(
    //     &[2, 4, 4][..],
    //     (0..2 * 4 * 4).map(|idx| random()).collect::<Vec<f32>>(),
    // )
    // .unwrap();
    // let mut input_grad = ArrayD::<f32>::zeros(input.shape());

    // println!("{}", input);
    // println!("{}", input_grad);

    // let max_pooling = MaxPooling2DNonBatch::new(1, 2);
    // let out = max_pooling.forward(input.view()).unwrap();
    // let out_grad = ArrayD::<f32>::ones(out.shape());
    // println!("\n{}", out);

    // max_pooling
    //     .backpropagation(input.view(), input_grad.view_mut(), out_grad.view())
    //     .unwrap();

    // println!("{}", input_grad);

    // let mut conv2d = Conv2DNonBatch::new(3, 3, 2);

    // let out = conv2d.forward(input.view());
    // let out_grad = ArrayD::<f32>::ones(out.shape());

    // println!("out");
    // println!("{}\n", out);
    // println!("out grad");
    // println!("{}\n", out_grad);

    // conv2d.backpropagation(input.view(), &mut input_grad, out_grad.view());
    // println!("input grad");
    // println!("{}", input_grad);
}
