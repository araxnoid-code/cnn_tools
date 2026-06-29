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

pub fn draw_plot(caption: &str, series: Vec<(f32, f32)>, path: &str, size: (u32, u32)) {
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
