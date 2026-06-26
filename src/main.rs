use cnn_tools::{draw_plot, get_train_dataset, traint_model};

fn main() {
    let dataset = get_train_dataset();
    let loss = traint_model(dataset);
    draw_plot("training", loss, "training.png", (750, 750));
}
