use ndarray::{ArrayViewD, s};

pub struct MaxPooling2DNonBatch {
    stride: usize,
    pool_size: usize,
}

impl MaxPooling2DNonBatch {
    pub fn new(stride: usize, pool_size: usize) -> MaxPooling2DNonBatch {
        Self { stride, pool_size }
    }

    pub fn forward(&self, input: ArrayViewD<f32>) -> Result<(), &str> {
        let input_shape = input.shape();
        if input_shape.len() != 3 {
            return Err("ukuran input untuk MaxPooling2DNonBatch haruslah 3D");
        } else if input_shape[1] != input_shape[2] {
            return Err("untuk saat ini, ukuran input haruslah sama untuk width dan high nya");
        }

        let channel_size = input_shape[0];
        let out_size = ((input_shape[1] - self.pool_size) / self.stride) + 1;

        for channel_idx in 0..channel_size {
            for row in (0..out_size).step_by(self.stride) {
                for coll in (0..out_size).step_by(self.stride) {
                    let slice = input.slice(s![
                        channel_idx..channel_idx + 1,
                        row..row + self.pool_size,
                        coll..coll + self.pool_size
                    ]);

                    println!("====");
                    println!("{}", slice);
                    println!("====");
                }
            }
        }

        Ok(())
    }
}
