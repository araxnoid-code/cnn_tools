use cnn_tools::{draw_plot, get_test_dataset, get_train_dataset, model_load, model_train};

fn main() {
    let test_dataset = get_test_dataset();
    let loss = model_load(test_dataset);
    // draw_plot("eval", loss, "eval.png", (750, 750));
}
