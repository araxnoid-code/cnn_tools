use std::ops::AddAssign;

use ndarray::{ArrayBase, ArrayD, ArrayViewD, ArrayViewMutD, Axis, Dim, IxDynImpl, OwnedRepr};

pub struct Softmax {
    saved: Option<ArrayD<f32>>,
    axis: usize,
}

impl Softmax {
    pub fn new(axis: usize) -> Softmax {
        Self { axis, saved: None }
    }

    pub fn forward(&mut self, input: ArrayViewD<'_, f32>) -> Result<(), &str> {
        if self.axis > input.len() {
            return Err("softmax error");
        }

        let mut shape = input.shape().to_vec();
        shape[self.axis] = 1;

        let exp = input.exp();
        let sum = exp.sum_axis(Axis(self.axis));
        let denom = sum.to_shape(shape).unwrap();

        let result = &exp / &denom.view();
        self.saved = Some(result);

        Ok(())
    }

    pub fn get_ouput(&self) -> Option<ArrayBase<ndarray::ViewRepr<&f32>, Dim<IxDynImpl>, f32>> {
        if let Some(arr) = &self.saved {
            return Some(arr.view());
        }
        None
    }

    pub fn backward(
        &self,
        mut input_gradient: Option<ArrayViewMutD<f32>>,
        gradient: ArrayViewD<f32>,
    ) -> Result<(), &str> {
        if let Some(input_gradient) = &mut input_gradient {
            let y = self.saved.as_ref().ok_or("error")?.view();

            let gy_sum = (&y * &gradient)
                .sum_axis(Axis(self.axis))
                .insert_axis(Axis(self.axis));

            let grad_input = &y * (&gradient - &gy_sum);
            input_gradient.add_assign(&grad_input);
        }

        Ok(())
    }
}
