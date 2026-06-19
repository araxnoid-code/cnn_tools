use std::ops::AddAssign;

use ndarray::{ArrayBase, ArrayViewD, ArrayViewMutD, Dim, IxDynImpl, OwnedRepr};

pub fn relu(input: ArrayViewD<f32>) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>, f32> {
    input.mapv(|x| x.max(0.))
}

pub fn relu_backward(
    input: ArrayViewD<f32>,
    mut input_gradient: Option<ArrayViewMutD<f32>>,
    gradient: ArrayViewD<f32>,
) {
    if let Some(input_gradient) = &mut input_gradient {
        let d = input.mapv(|x| if x > 0. { 1. } else { 0. }) * gradient;
        input_gradient.add_assign(&d);
    }
}
