use std::{fs::read_dir, path::Path, thread::available_parallelism};

use cnn_tools::{
    Conv2DNonBatch, LinaerNonBatch, MaxPooling2DNonBatch, Softmax, cross_entropy_loss,
    cross_entropy_loss_backward, relu, relu_backward,
};
use image::imageops::FilterType;
use ndarray::{ArrayBase, ArrayD, Axis, Dim, IxDynImpl, OwnedRepr, array};
use plotters::{
    backend::BitMapBackend,
    chart::ChartBuilder,
    drawing::IntoDrawingArea,
    element::{Circle, EmptyElement},
    series::PointSeries,
    style::{
        IntoFont, ShapeStyle,
        full_palette::{BLACK, RED, WHITE},
    },
};

fn draw_plot(caption: &str, series: Vec<(f32, f32)>, path: &str, size: (u32, u32)) {
    let x_max = series
        .iter()
        .max_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
        .unwrap();
    let y_max = series
        .iter()
        .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
        .unwrap();

    let root = BitMapBackend::new(path, size).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 25).into_font())
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f32..x_max.0, 0f32..y_max.1)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series(PointSeries::of_element(series, 5, &RED, &|c, _, _| {
            return EmptyElement::at(c) + Circle::new((0, 0), 3, ShapeStyle::from(&BLACK).filled());
        }))
        .unwrap();

    root.present().unwrap();
}

fn main() {
    let data = vec![
        (0., 10.),
        (10., 20.),
        (8., 7.),
        (8., 3.),
        (9., 13.),
        (1., 6.),
    ];

    draw_plot("data", data, "testing.png", (750, 750));

    // let dataset = get_dataset();
    // model(dataset);
}

fn get_dataset() -> Vec<(
    ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>, f32>,
    ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>, f32>,
)> {
    let mut batch_sample_train = vec![];
    let train_path_glasses = Path::new("train/glasses");
    let read_path = read_dir(train_path_glasses).expect("The Path Not Found");
    for entry in read_path {
        match entry {
            Ok(entry) => {
                let file_name = entry.file_name().into_string().unwrap();

                let img =
                    image::open(format!("train/glasses/{}", file_name)).expect("image not found");

                let pixels = img.resize_exact(64, 64, FilterType::Nearest).into_rgb32f();

                let mut red = Vec::with_capacity(64);
                let mut green = Vec::with_capacity(64);
                let mut blue = Vec::with_capacity(64);
                for rgb in pixels.pixels() {
                    let value = rgb.0;
                    red.push(value[0] / 1.);
                    green.push(value[1] / 1.);
                    blue.push(value[2] / 1.);
                }
                let rgb_vec = [red, green, blue].concat();
                let array = ArrayD::<f32>::from_shape_vec(vec![3, 64, 64], rgb_vec).unwrap();
                let label = ArrayD::<f32>::from_shape_vec(vec![1, 2], vec![1., 0.]).unwrap();

                batch_sample_train.push((array, label));
            }
            Err(e) => println!("error in entry, {}", e),
        }
    }

    let train_path_glasses = Path::new("train/noglasses");
    let read_path = read_dir(train_path_glasses).expect("The Path Not Found");
    for entry in read_path {
        match entry {
            Ok(entry) => {
                let file_name = entry.file_name().into_string().unwrap();

                let img =
                    image::open(format!("train/noglasses/{}", file_name)).expect("image not found");

                let pixels = img.resize_exact(64, 64, FilterType::Nearest).into_rgb32f();

                let mut red = Vec::with_capacity(64);
                let mut green = Vec::with_capacity(64);
                let mut blue = Vec::with_capacity(64);
                for rgb in pixels.pixels() {
                    let value = rgb.0;
                    red.push(value[0] / 1.);
                    green.push(value[1] / 1.);
                    blue.push(value[2] / 1.);
                }
                let rgb_vec = [red, green, blue].concat();
                let array = ArrayD::<f32>::from_shape_vec(vec![3, 64, 64], rgb_vec).unwrap();
                let label = ArrayD::<f32>::from_shape_vec(vec![1, 2], vec![0., 1.]).unwrap();

                batch_sample_train.push((array, label));
            }
            Err(e) => println!("error in entry, {}", e),
        }
    }
    batch_sample_train
}

fn model(batch: Vec<(ArrayD<f32>, ArrayD<f32>)>) {
    // model
    let mut conv2d_1 = Conv2DNonBatch::new(3, 8, 3);
    let max_pooling_1 = MaxPooling2DNonBatch::new(2, 2);

    let mut conv2d_2 = Conv2DNonBatch::new(8, 16, 3);
    let max_pooling_2 = MaxPooling2DNonBatch::new(2, 2);

    let mut conv2d_3 = Conv2DNonBatch::new(16, 32, 3);
    let max_pooling_3 = MaxPooling2DNonBatch::new(2, 2);

    let mut linear_1 = LinaerNonBatch::new(1152, 512);
    let mut linear_2 = LinaerNonBatch::new(512, 2);
    let mut softmax = Softmax::new(1);

    // model

    for epoch in 0..32 {
        let mut mean = 0.;
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
            let prop = softmax.get_ouput().unwrap();

            let loss = cross_entropy_loss(prop.view(), label.view(), 1);
            mean += loss[[0, 0]];

            // backpropagation
            let mut conv2d_1_result_gradient = ArrayD::<f32>::zeros(conv2d_1_result.shape());
            let mut relu_1_gradient = ArrayD::<f32>::zeros(relu_1.shape());
            let mut max_pooling_1_result_gradient =
                ArrayD::<f32>::zeros(max_pooling_1_result.shape());

            let mut conv2d_2_result_gradient = ArrayD::<f32>::zeros(conv2d_2_result.shape());
            let mut relu_2_gradient = ArrayD::<f32>::zeros(relu_2.shape());
            let mut max_pooling_2_result_gradient =
                ArrayD::<f32>::zeros(max_pooling_2_result.shape());

            let mut conv2d_3_result_gradient = ArrayD::<f32>::zeros(conv2d_3_result.shape());
            let mut relu_conv2d_3_gradient = ArrayD::<f32>::zeros(relu_conv2d_3.shape());
            let max_pooling_3_result_gradient;

            let mut flatten_gradient = ArrayD::<f32>::zeros(flatten.shape());
            let mut relu_3_gradient = ArrayD::<f32>::zeros(relu_3.shape());
            let mut linear_1_result_gradient = ArrayD::<f32>::zeros(linear_1_result.shape());
            let mut relu_res_gradient = ArrayD::<f32>::zeros(relu_res.shape());
            let mut linear_2_result_gradient = ArrayD::<f32>::zeros(linear_2_result.shape());
            let mut gradient_prop = ArrayD::<f32>::zeros(prop.shape());

            let loss_gradient = ArrayD::<f32>::ones(loss.shape());
            cross_entropy_loss_backward(
                prop.view(),
                Some(gradient_prop.view_mut()),
                label.view(),
                None,
                loss_gradient.view(),
            );
            softmax
                .backward(
                    Some(linear_2_result_gradient.view_mut()),
                    gradient_prop.view(),
                )
                .unwrap();
            linear_2.backward(
                relu_res.view(),
                Some(relu_res_gradient.view_mut()),
                linear_2_result_gradient.view(),
            );
            relu_backward(
                linear_1_result.view(),
                Some(linear_1_result_gradient.view_mut()),
                relu_res_gradient.view(),
            );
            linear_1.backward(
                relu_3.view(),
                Some(relu_3_gradient.view_mut()),
                linear_1_result_gradient.view(),
            );
            relu_backward(
                flatten.view(),
                Some(flatten_gradient.view_mut()),
                relu_3_gradient.view(),
            );

            max_pooling_3_result_gradient = flatten_gradient
                .to_shape(max_pooling_3_result.shape())
                .unwrap()
                .to_owned();

            max_pooling_3
                .backward(
                    relu_conv2d_3.view(),
                    Some(relu_conv2d_3_gradient.view_mut()),
                    max_pooling_3_result_gradient.view(),
                )
                .unwrap();

            relu_backward(
                conv2d_3_result.view(),
                Some(conv2d_3_result_gradient.view_mut()),
                relu_conv2d_3_gradient.view(),
            );

            conv2d_3.backward(
                max_pooling_2_result.view(),
                Some(max_pooling_2_result_gradient.view_mut()),
                conv2d_3_result_gradient.view(),
            );

            //
            max_pooling_2
                .backward(
                    relu_2.view(),
                    Some(relu_2_gradient.view_mut()),
                    max_pooling_2_result_gradient.view(),
                )
                .unwrap();

            relu_backward(
                conv2d_2_result.view(),
                Some(conv2d_2_result_gradient.view_mut()),
                relu_2_gradient.view(),
            );

            conv2d_2.backward(
                max_pooling_1_result.view(),
                Some(max_pooling_1_result_gradient.view_mut()),
                conv2d_2_result_gradient.view(),
            );

            max_pooling_1
                .backward(
                    relu_1.view(),
                    Some(relu_1_gradient.view_mut()),
                    max_pooling_1_result_gradient.view(),
                )
                .unwrap();

            relu_backward(
                conv2d_1_result.view(),
                Some(conv2d_1_result_gradient.view_mut()),
                relu_1_gradient.view(),
            );

            conv2d_1.backward(sample.view(), None, conv2d_1_result_gradient.view());

            // optim
            let lr = 0.0001;
            conv2d_1.adam_optim(lr);
            conv2d_2.adam_optim(lr);
            conv2d_3.adam_optim(lr);
            linear_1.adam_optim(lr);
            linear_2.adam_optim(lr);

            // zero grad
            conv2d_1.zero_grad();
            conv2d_2.zero_grad();
            conv2d_3.zero_grad();
            linear_1.zero_grad();
            linear_2.zero_grad();
        }
        println!("epoch {}, mean loss: {}", epoch, mean / batch.len() as f32);
    }
}
