use gomokugen::board::Board;
use kn_graph::{onnx, optimizer, dtype::{DTensor, Tensor}, ndarray::IxDyn};

fn main() {
    // // Load on onnx file into a graph
    // let graph = load_graph_from_onnx_path("test.onnx", false)?;
    // // Optimize the graph
    // let graph = optimize_graph(&graph, Default::default());
    // // Render the graph as an svg file
    // graph_to_svg("test.svg", &graph, false, false)?;

    // // Build the inputs
    // let batch_size = 8;
    // let inputs = [DTensor::F32(Tensor::zeros(IxDyn(&[batch_size, 16])))];

    // // CPU:
    // // just evaluate the graph
    // let outputs: Vec<DTensor> = cpu_eval_graph(&graph, batch_size, &inputs);

    // Load an onnx file into a Graph.
    let graph = onnx::load_graph_from_onnx_path("./model.onnx", false).unwrap();
    // Optimise the graph.
    let graph = optimizer::optimize_graph(&graph, Default::default());
    // Render the graph as an svg.
    kn_graph::dot::graph_to_svg("./model.svg", &graph, false, false).unwrap();

    let board = Board::new();

    let outputs = generate_policy(&graph, &board);

    display_net_output(&outputs);
}

fn generate_policy(graph: &kn_graph::graph::Graph, board: &Board<9>) -> Vec<DTensor> {
    // build inputs
    let batch_size = 1;
    // inputs are a 162 1-D element vector
    let inputs = [DTensor::F32(Tensor::zeros(IxDyn(&[batch_size, 162])))];

    // evaluate the graph on this input
    kn_graph::cpu::cpu_eval_graph(graph, batch_size, &inputs)
}

fn display_net_output(outputs:&[DTensor]) {
    let [DTensor::F32(output)] = outputs else {
        panic!("Expected a single output tensor of f32");
    };

    let output = output.as_slice().unwrap();

    // print the output as a 9x9 image
    let min = output.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = output.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let rendering_function = |val: f32| {
        const ASCII_BRIGHTNESS: &str = " .:-=+*#%@";
        let val = (val - min) / (max - min);
        let idx = (val * (ASCII_BRIGHTNESS.len() - 1) as f32).round() as usize;
        ASCII_BRIGHTNESS.as_bytes()[idx] as char
    };
    for i in 0..9 {
        for j in 0..9 {
            print!("{c}{c}", c = rendering_function(output[i * 9 + j]));
        }
        println!();
    }
}
