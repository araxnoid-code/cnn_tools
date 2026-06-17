use std::ops::AddAssign;

use ndarray::{
    Array2, ArrayBase, ArrayD, ArrayViewD, ArrayViewMutD, Axis, Dim, IxDynImpl, OwnedRepr,
    linalg::Dot,
};

pub struct LinaerNonBatch {
    weight: ArrayD<f32>,
    gradient_weight: ArrayD<f32>,
    bias: ArrayD<f32>,
    gradient_bias: ArrayD<f32>,
}

impl LinaerNonBatch {
    pub fn new(in_feature: usize, out_feature: usize) -> LinaerNonBatch {
        Self {
            weight: ArrayD::<f32>::from_shape_vec(
                vec![in_feature, out_feature],
                (0..in_feature * out_feature)
                    .map(|idx| idx as f32)
                    .collect::<Vec<f32>>(),
            )
            .unwrap(),
            gradient_weight: ArrayD::<f32>::zeros(vec![in_feature, out_feature]),
            bias: ArrayD::<f32>::from_shape_vec(
                vec![out_feature],
                (0..out_feature).map(|idx| idx as f32).collect::<Vec<f32>>(),
            )
            .unwrap(),
            gradient_bias: ArrayD::<f32>::zeros(vec![out_feature]),
        }
    }

    pub fn backward(
        &mut self,
        input: ArrayViewD<f32>,
        mut input_grad: ArrayViewMutD<f32>,
        gradient: ArrayViewD<f32>,
    ) {
        let d_input = gradient.dot(&self.weight.t());
        input_grad.add_assign(&d_input);

        let d_weight = input.t().dot(&gradient);
        self.gradient_weight.add_assign(&d_weight);

        let d_bias = gradient.sum_axis(Axis(0));
        self.gradient_bias.add_assign(&d_bias);
    }

    pub fn forward(
        &self,
        input: ArrayViewD<f32>,
    ) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>, f32> {
        input.dot(&self.weight.view()) + &self.bias
    }
}
