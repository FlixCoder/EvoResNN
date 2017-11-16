extern crate ernn;

use ernn::*;


#[test]
fn xor()
{
	//parameters for genetic optimization
	let population = 25;
	let prob_block = 0.1;
	let prob_op = 0.5;
	let op_range = 0.5;
	
	// create examples of the xor function
	let examples = [
		(vec![0f64, 0f64], vec![0f64]),
		(vec![0f64, 1f64], vec![1f64]),
		(vec![1f64, 0f64], vec![1f64]),
		(vec![1f64, 1f64], vec![0f64]),
	];
	
    // create a new neural network, evaluator and optimizer
    let nn = NN::new(2, 3, 1, Activation::Quadratic, Activation::Sigmoid);
	let eval = XorEval::new();
	let mut opt = Optimizer::new(eval, nn);
	//generate initial population
	let mut mse = -opt.gen_population(5);
	println!("initial MSE: {}", mse);
	
    // train the network
    while mse > 0.01
	{
		mse = -opt.optimize_easy(10, population, prob_block, prob_op, op_range);
		println!("MSE: {}", mse);
	}
	let nn = opt.get_nn(); //get the best neural net

    // make sure json encoding/decoding works as expected
    let json = nn.to_json();
    let nn2 = NN::from_json(&json);

    // test the trained network
    for &(ref inputs, ref outputs) in examples.iter()
	{
        let results = nn2.run(inputs);
        let (result, key) = (results[0], outputs[0]);
        assert!((result - key).abs() < 0.1);
    }
}


struct XorEval
{
	examples: Vec<(Vec<f64>, Vec<f64>)>,
}

impl XorEval
{
	fn new() -> XorEval
	{
		// create examples of the xor function
		let examples = vec![
			(vec![0f64, 0f64], vec![0f64]),
			(vec![0f64, 1f64], vec![1f64]),
			(vec![1f64, 0f64], vec![1f64]),
			(vec![1f64, 1f64], vec![0f64]),
		];
		XorEval{ examples: examples }
	}
}

//implement an evaluator to rate the neural nets results
impl Evaluator for XorEval
{
	fn evaluate(&self, nn:&NN) -> f64
	{ //optimize the mean squared error
		let mut sum = 0.0;
		for &(ref inputs, ref outputs) in self.examples.iter()
		{
			let results = nn.run(inputs);
			let diff = results[0] - outputs[0];
			sum += diff * diff;
		}
		sum /= self.examples.len() as f64;
		-sum //higher is good, so return minus mean squared error
	}
}
