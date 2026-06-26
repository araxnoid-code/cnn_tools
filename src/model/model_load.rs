use ndarray::{ArrayD, Axis};

use crate::{
    Conv2DNonBatch, LinaerNonBatch, MaxPooling2DNonBatch, Softmax, cross_entropy_loss, relu,
};

pub fn model_load(batch: Vec<(ArrayD<f32>, ArrayD<f32>)>) -> Vec<(f32, f32)> {
    let mut loss_save = vec![];
    // model
    let mut conv2d_1 = Conv2DNonBatch::new(3, 8, 3);
    conv2d_1.load_params("params/conv2d_1.json");

    let max_pooling_1 = MaxPooling2DNonBatch::new(2, 2);

    let mut conv2d_2 = Conv2DNonBatch::new(8, 16, 3);
    conv2d_2.load_params("params/conv2d_2.json");

    let max_pooling_2 = MaxPooling2DNonBatch::new(2, 2);

    let mut conv2d_3 = Conv2DNonBatch::new(16, 32, 3);
    conv2d_3.load_params("params/conv2d_3.json");

    let max_pooling_3 = MaxPooling2DNonBatch::new(2, 2);

    let mut linear_1 = LinaerNonBatch::new(1152, 512);
    linear_1.load_params("params/linear_1.json");

    let mut linear_2 = LinaerNonBatch::new(512, 2);
    linear_2.load_params("params/linear_2.json");

    let mut softmax = Softmax::new(1);
    // model

    for (idx, (sample, label)) in batch.iter().enumerate() {
        let conv2d_1_result = conv2d_1.forward(sample.view());
        let relu_1 = relu(conv2d_1_result.view());
        let max_pooling_1_result = max_pooling_1.forward(relu_1.view()).unwrap();

        let conv2d_2_result = conv2d_2.forward(max_pooling_1_result.view());
        let relu_2 = relu(conv2d_2_result.view());
        let max_pooling_2_result = max_pooling_2.forward(relu_2.view()).unwrap();

        let conv2d_3_result = conv2d_3.forward(max_pooling_2_result.view());
        let relu_conv2d_3 = relu(conv2d_3_result.view());
        let max_pooling_3_result = max_pooling_3.forward(relu_conv2d_3.view()).unwrap();

        let flatten = max_pooling_3_result
            .flatten()
            .into_dyn()
            .insert_axis(Axis(0));

        let relu_3 = relu(flatten.view());

        let linear_1_result = linear_1.forward(relu_3.view());

        let relu_res = relu(linear_1_result.view());

        let linear_2_result = linear_2.forward(relu_res.view());

        softmax.forward(linear_2_result.view()).unwrap();
        let predict = softmax.get_ouput().unwrap();

        let loss = cross_entropy_loss(predict.view(), label.view(), 1);

        let predict_flat = predict.flatten().to_vec();
        let predict_index = if predict_flat[0] > predict_flat[1] {
            0
        } else {
            1
        };

        let label_flat = label.flatten().to_vec();
        let label_index = if label_flat[0] > label_flat[1] { 0 } else { 1 };

        println!(
            "loss: {} | predict: {} | label: {} | status: {}",
            loss,
            predict_index,
            label_index,
            label_index == predict_index
        );
        loss_save.push((idx as f32, loss[[0, 0]]));
    }
    loss_save
}
