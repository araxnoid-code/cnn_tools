use std::ops::AddAssign;

use ndarray::{
    Array2, ArrayBase, ArrayD, ArrayViewD, ArrayViewMutD, Axis, Dim, IxDynImpl, OwnedRepr,
    linalg::Dot,
};
use rand::rng;
use rand_distr::{Distribution, Normal, StandardNormal, num_traits::Pow};

pub struct LinaerNonBatch {
    in_feature: usize,
    out_feature: usize,
    weight: ArrayD<f32>,
    gradient_weight: ArrayD<f32>,
    bias: ArrayD<f32>,
    gradient_bias: ArrayD<f32>,

    e: f32,
    t: usize,
    v_weight: ArrayD<f32>,
    m_weight: ArrayD<f32>,
    v_bias: ArrayD<f32>,
    m_bias: ArrayD<f32>,
    b_1: f32,
    b_2: f32,
}

impl LinaerNonBatch {
    pub fn adam_optim(&mut self, lr: f32) {
        // weight
        self.m_weight = &self.m_weight * self.b_1 + (1. - self.b_1) * &self.gradient_weight;
        let m_correction = &self.m_weight / (1. - self.b_1.powi(self.t as i32));

        self.v_weight = &self.v_weight * self.b_2 + (1. - self.b_2) * self.gradient_weight.pow2();
        let v_correction = &self.v_weight / (1. - self.b_2.powi(self.t as i32));

        self.weight = &self.weight - lr / (self.e + v_correction.sqrt()) * m_correction;

        // bias
        self.m_bias = &self.m_bias * self.b_1 + (1. - self.b_1) * &self.gradient_bias;
        let m_correction = &self.m_bias / (1. - self.b_1.powi(self.t as i32));

        self.v_bias = &self.v_bias * self.b_2 + (1. - self.b_2) * self.gradient_bias.pow2();
        let v_correction = &self.v_bias / (1. - self.b_2.powi(self.t as i32));

        self.bias = &self.bias - lr / (self.e + v_correction.sqrt()) * m_correction;

        self.t += 1;
    }

    pub fn zero_grad(&mut self) {
        self.gradient_weight = ArrayD::zeros(vec![self.in_feature, self.out_feature]);

        self.gradient_bias = ArrayD::<f32>::zeros(vec![self.out_feature]);
    }

    pub fn new(in_feature: usize, out_feature: usize) -> LinaerNonBatch {
        // He Init
        let std = (2.0 / in_feature as f32).sqrt();
        let normal = Normal::new(0., std).unwrap();
        let mut rng = rng();

        Self {
            in_feature,
            out_feature,
            weight: ArrayD::<f32>::from_shape_vec(
                vec![in_feature, out_feature],
                (0..in_feature * out_feature)
                    .map(|_| normal.sample(&mut rng))
                    .collect::<Vec<f32>>(),
            )
            .unwrap(),
            gradient_weight: ArrayD::<f32>::zeros(vec![in_feature, out_feature]),
            bias: ArrayD::<f32>::zeros(vec![out_feature]),
            gradient_bias: ArrayD::<f32>::zeros(vec![out_feature]),

            t: 1,
            e: 0.00000001,
            b_1: 0.9,
            b_2: 0.999,
            m_weight: ArrayD::<f32>::zeros(vec![in_feature, out_feature]),
            v_weight: ArrayD::<f32>::zeros(vec![in_feature, out_feature]),
            m_bias: ArrayD::<f32>::zeros(vec![out_feature]),
            v_bias: ArrayD::<f32>::zeros(vec![out_feature]),
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
