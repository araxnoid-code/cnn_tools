use cnn_tools::{draw_plot, get_test_dataset, get_train_dataset, model_load, model_train};

fn main() {
    // let train_dataset = get_train_dataset();
    let test_dataset = get_test_dataset();

    // let loss = model_train(train_dataset);
    // draw_plot("training", loss, "training.png", (750, 750));

    // let test_dataset = get_test_dataset();
    let loss = model_load(test_dataset);
    draw_plot("eval", loss, "eval.png", (750, 750));
}
