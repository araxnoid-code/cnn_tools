use std::iter::Sum;

use ndarray::{Array2, ArrayBase, ArrayD, Axis, Dim, IxDynImpl, OwnedRepr, Slice, s};

fn main() {
    let input = ArrayD::<f32>::from_shape_vec(
        &[3, 8, 8][..],
        (0..3 * 8 * 8)
            .map(|idx| idx as f32 * 0.1)
            .collect::<Vec<f32>>(),
    )
    .unwrap();

    println!("{}", input);

    let in_channel = 3;
    let out_channel = 4;
    let kernel_size = 3;
    let kernels = kernel_initial(in_channel, out_channel, kernel_size);

    let channel_size = input.shape()[0];
    let matrix_size = input.shape()[1];

    let mut out = vec![];
    for out_kernel in &kernels {
        //
        for row in 0..matrix_size - kernel_size + 1 {
            for coll in 0..matrix_size - kernel_size + 1 {
                //
                let mut acc: Option<ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>, f32>> = None;
                for channel_idx in 0..channel_size {
                    // [1, 8, 8]
                    let matrix = input.slice_axis(
                        Axis(0),
                        Slice::new(channel_idx as isize, Some(channel_idx as isize + 1), 1),
                    );

                    let kernel = &out_kernel[channel_idx];

                    let matrix = matrix.to_shape(&[8, 8][..]).unwrap();
                    let slice = matrix.slice(s![row..row + kernel_size, coll..coll + kernel_size]);

                    let result = slice.dot(kernel);
                    if let Some(acc) = &mut acc {
                        *acc = &*acc + result;
                    } else {
                        acc = Some(result)
                    }
                }
                let sum = acc.unwrap().sum();
                out.push(sum);
            }
        }
    }
    let output = ArrayD::<f32>::from_shape_vec(
        &[
            out_channel,
            matrix_size - kernel_size + 1,
            matrix_size - kernel_size + 1,
        ][..],
        out,
    )
    .unwrap();

    println!("{}", output);

    // prediction output shape is [4, 6, 6]
}

fn kernel_initial(
    in_channel: usize,
    out_channel: usize,
    kernel: usize,
) -> Vec<Vec<ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>, f32>>> {
    let mut kernels = vec![];
    for _ in 0..out_channel {
        let mut out_kernel = vec![];
        for _ in 0..in_channel {
            let kernel = Array2::<f32>::from_shape_vec(
                [kernel, kernel],
                (0..kernel * kernel)
                    .map(|idx| idx as f32 * 0.001)
                    .collect::<Vec<f32>>(),
            )
            .unwrap();
            out_kernel.push(kernel);
        }
        kernels.push(out_kernel);
    }
    kernels
}
