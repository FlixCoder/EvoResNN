extern crate ernn;

use ernn::*;


fn main()
{
	//parameters for genetic optimization
	let population = 100; //the bigger the network, the bigger this. however, this affects performance very hard, but stabilizes learning
	let prob_block = 0.02; //can be adjusted, so the network does not adds too many layers or adds more layers (no added blocks probably means prob_op or op_range too high)
	let prob_op = 0.75; //the bigger the network, the lower this (try op_range first)
	let op_range = 0.25; //the bigger the network, the lower this (lower this before prob_op)
	
    // create a new neural network, evaluator and optimizer
	let nn = NN::new(2, 100, 1, Activation::LRELU, Activation::Linear);
	let eval = XorEval::new();
	let mut opt = Optimizer::new(eval, nn);
	//generate initial population
	let mut mse = -opt.gen_population(10);
	
    // train the network
	let mut i = 0;
    while mse > 0.01
	{
		mse = -opt.optimize_easy_par(10, population, prob_block, prob_op, op_range);
		println!("MSE: {}", mse);
		i += 10;
	}
	
	let nn = opt.get_nn(); //get the best neural net
	println!("NN information:");
	println!("Iterations: {}", i);
	println!("Generation: {}", nn.get_gen());
	println!("Added blocks: {}", nn.get_blocks());
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
			let mut result = 0.0;
			const N:u16 = 100;
			for _ in 1..N
			{ // just as simple example of dropout usage
				let results = nn.run_dropout(inputs, 0.5);
				result += results[0] as f64;
			}
			result /= N as f64;
			let diff = result - outputs[0];
			sum += diff * diff;
		}
		sum /= self.examples.len() as f64;
		-sum //higher is good, so return minus mean squared error
	}
}
