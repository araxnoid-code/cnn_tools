use std::{fs::read_dir, path::Path};

use image::imageops::FilterType;
use ndarray::{ArrayBase, ArrayD, Dim, IxDynImpl, OwnedRepr};

pub fn get_test_dataset() -> Vec<(
    ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>, f32>,
    ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>, f32>,
)> {
    let mut batch_sample_test = vec![];
    let test_path_glasses = Path::new("test/glasses");
    let read_path = read_dir(test_path_glasses).expect("The Path Not Found");
    for entry in read_path {
        match entry {
            Ok(entry) => {
                let file_name = entry.file_name().into_string().unwrap();

                let img =
                    image::open(format!("test/glasses/{}", file_name)).expect("image not found");

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

                batch_sample_test.push((array, label));
            }
            Err(e) => println!("error in entry, {}", e),
        }
    }

    let test_path_glasses = Path::new("test/noglasses");
    let read_path = read_dir(test_path_glasses).expect("The Path Not Found");
    for entry in read_path {
        match entry {
            Ok(entry) => {
                let file_name = entry.file_name().into_string().unwrap();

                let img =
                    image::open(format!("test/noglasses/{}", file_name)).expect("image not found");

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

                batch_sample_test.push((array, label));
            }
            Err(e) => println!("error in entry, {}", e),
        }
    }
    batch_sample_test
}
