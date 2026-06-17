use ndarray::{ArrayBase, ArrayViewD, Dim, IxDynImpl, OwnedRepr};

pub struct Exp {}

impl Exp {
    pub fn new() -> Exp {
        Self {}
    }

    pub fn forward(
        &self,
        input: ArrayViewD<f32>,
    ) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>, f32> {
        input.exp()
    }

    pub fn backward(
        &self,
        input: ArrayViewD<f32>,
        gradient: ArrayViewD<f32>,
    ) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>, f32> {
        input.exp() * gradient
    }
}
