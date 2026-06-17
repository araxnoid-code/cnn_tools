use ndarray::{ArrayViewD, Axis};

pub fn softmax(input: ArrayViewD<'_, f32>, axis: usize) -> Result<(), &str> {
    if axis > input.len() {
        return Err("softmax error");
    }

    let mut shape = input.shape().to_vec();
    shape[axis] = 1;

    let exp = input.exp();
    let sum = exp.sum_axis(Axis(axis));
    let denom = sum.to_shape(shape).unwrap();
    println!("{}", denom);
    let result = &exp / &denom.view();
    println!("{}", result);

    Ok(())
}
