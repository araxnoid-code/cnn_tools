use std::ops::AddAssign;

use ndarray::{
    Array2, ArrayBase, ArrayD, ArrayViewD, ArrayViewMutD, Axis, Dim, IxDynImpl, OwnedRepr,
    linalg::Dot,
};
use rand::rng;
use rand_distr::{Distribution, Normal, StandardNormal};

pub struct LinaerNonBatch {
    weight: ArrayD<f32>,
    gradient_weight: ArrayD<f32>,
    bias: ArrayD<f32>,
    gradient_bias: ArrayD<f32>,
}

impl LinaerNonBatch {
    pub fn new(in_feature: usize, out_feature: usize) -> LinaerNonBatch {
        // He Init
        let std = (2.0 / in_feature as f32).sqrt();
        let normal = Normal::new(0., std).unwrap();
        let mut rng = rng();

        Self {
            weight: ArrayD::<f32>::from_shape_vec(
                vec![in_feature, out_feature],
                (0..in_feature * out_feature)
                    .map(|_| normal.sample(&mut rng))
                    .collect::<Vec<f32>>(),
            )
            .unwrap(),
            gradient_weight: ArrayD::<f32>::zeros(vec![in_feature, out_feature]),
            bias: ArrayD::<f32>::from_shape_vec(
                vec![out_feature],
                (0..out_feature)
                    .map(|_| normal.sample(&mut rng))
                    .collect::<Vec<f32>>(),
            )
            .unwrap(),
            gradient_bias: ArrayD::<f32>::zeros(vec![out_feature]),
        }
    }

    pub fn backward(
        &mut self,
        input: ArrayViewD<f32>,
        mut input_grad: Option<ArrayViewMutD<f32>>,
        gradient: ArrayViewD<f32>,
    ) {
        if let Some(input_grad) = &mut input_grad {
            let d_input = gradient.dot(&self.weight.t());
            input_grad.add_assign(&d_input);
        }

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
