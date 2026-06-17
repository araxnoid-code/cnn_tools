use std::ops::AddAssign;

use ndarray::{ArrayBase, ArrayViewD, ArrayViewMutD, Axis, Dim, IxDynImpl, OwnedRepr};

pub struct Softmax {
    axis: usize,
}

impl Softmax {
    pub fn new(axis: usize) -> Softmax {
        Self { axis }
    }

    pub fn forward(
        &self,
        input: ArrayViewD<'_, f32>,
    ) -> Result<ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>, f32>, &str> {
        if self.axis > input.len() {
            return Err("softmax error");
        }

        let mut shape = input.shape().to_vec();
        shape[self.axis] = 1;

        let exp = input.exp();
        let sum = exp.sum_axis(Axis(self.axis));
        let denom = sum.to_shape(shape).unwrap();

        let result = &exp / &denom.view();

        Ok(result)
    }

    pub fn backward(
        &self,
        input: ArrayViewD<f32>,
        mut input_gradient: ArrayViewMutD<f32>,
        gradient: ArrayViewD<f32>,
    ) -> Result<(), &str> {
        let y = self.forward(input)?;

        let gy_sum = (&y * &gradient)
            .sum_axis(Axis(self.axis))
            .insert_axis(Axis(self.axis));

        let grad_input = &y * (&gradient - &gy_sum);
        input_gradient.add_assign(&grad_input);

        Ok(())
    }
}
