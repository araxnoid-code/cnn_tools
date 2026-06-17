use std::ops::AddAssign;

use ndarray::{ArrayBase, ArrayViewD, ArrayViewMutD, Axis, Dim, IxDynImpl, OwnedRepr};

pub fn softmax(
    input: ArrayViewD<'_, f32>,
    axis: usize,
) -> Result<ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>, f32>, &'static str> {
    if axis > input.len() {
        return Err("softmax error");
    }

    let mut shape = input.shape().to_vec();
    shape[axis] = 1;

    let exp = input.exp();
    let sum = exp.sum_axis(Axis(axis));
    let denom = sum.to_shape(shape).unwrap();

    let result = &exp / &denom.view();
    Ok(result)
}

pub fn softmax_backward(
    input: ArrayViewD<'_, f32>,
    mut input_gradient: ArrayViewMutD<'_, f32>,
    axis: usize,
    gradient: ArrayViewD<f32>,
) -> Result<(), &'static str> {
    let y = softmax(input, axis)?;

    let d = (&y.view() * &gradient)
        .sum_axis(Axis(axis))
        .insert_axis(Axis(axis));

    let d = y * (&gradient - &d.view());
    input_gradient.add_assign(&d);

    Ok(())
}
