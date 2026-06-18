use std::ops::AddAssign;

use ndarray::{
    ArrayBase, ArrayViewD, ArrayViewMut, ArrayViewMutD, Axis, Dim, IxDynImpl, OwnedRepr,
};

const CROSS_ENTROPY_LOSS_E: f32 = 0.00001;

pub fn cross_entropy_loss(
    pred: ArrayViewD<f32>,
    actual: ArrayViewD<f32>,
    axis: usize,
) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>, f32> {
    let loss = -(&actual * (&pred + CROSS_ENTROPY_LOSS_E).ln())
        .sum_axis(Axis(axis))
        .insert_axis(Axis(axis));
    loss
}

pub fn cross_entropy_loss_backward(
    pred: ArrayViewD<f32>,
    mut pred_gradient: Option<ArrayViewMutD<f32>>,
    actual: ArrayViewD<f32>,
    mut actual_gradient: Option<ArrayViewMutD<f32>>,
    gradient: ArrayViewD<f32>,
) {
    if let Some(pred_grad) = &mut pred_gradient {
        let d_pred = -(&actual / &pred) * &gradient;
        pred_grad.add_assign(&d_pred);
    }

    if let Some(actual_grad) = &mut actual_gradient {
        let d_actual = -(&pred + CROSS_ENTROPY_LOSS_E).ln() * gradient;
        actual_grad.add_assign(&d_actual);
    }
}
