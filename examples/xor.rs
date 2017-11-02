extern crate ernn;

use ernn::*;


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


fn main()
{
	//parameters for genetic optimization
	let population = 50;
	let survival = 8;
	let badsurv = 2;
	let prob_avg = 0.1;
	let prob_mut = 0.9;
	let prob_op = 0.75;
	let op_range = 0.9;
	let prob_block = 0.05;
	let prob_new = 0.1;
	
    // create a new neural network, evaluator and optimizer
    let nn = NN::new(2, 3, 1, Activation::Quadratic, Activation::Sigmoid);
	let eval = XorEval::new();
	let mut opt = Optimizer::new(eval, nn);
	//generate initial population
	let mut mse = -opt.optimize(1, survival+badsurv, survival, badsurv, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
	
    // train the network
	let mut i = 0;
    while mse > 0.01
	{
		mse = -opt.optimize(10, population, survival, badsurv, prob_avg, prob_mut, prob_op, op_range, prob_block, prob_new);
		println!("MSE: {}", mse);
		i += 10;
	}
	
	let nn = opt.get_nn(); //get the best neural net
	println!("NN information:");
	println!("Iterations: {}", i);
	println!("Generation: {}", nn.get_gen());
	println!("Added blocks: {}", nn.get_blocks());
}
