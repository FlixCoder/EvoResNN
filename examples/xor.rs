extern crate ernn;

use ernn::*;


//model guide:
//not very sure, but currently LRELU as hidden-layer activation seems to perform best, especially for adding blocks.
//for some tasks however other activation functions might perform better.
//example: for sigmoid as output activation a quadratic hidden activation can lead to faster convergence to 0 and 1 values

//parameter guide:
//for parameter adjustment information see comment behind parameters
//in general: preferably try to lower hidden-layer-size if learning is unstable/slow
//higher population explores more directions to go to. affects performance very hard, but still is a good way to
//find convergence faster. prefers fast convergence, so might be a more compact model.
//if the model adds too many blocks, try increasing prob_op and op_range or decrease prob_block
//if the evaluator is noisy, you can try increasing survival, badsurv and population


fn main()
{
	//parameters for genetic optimization
	let population = 100; //the bigger the network, the bigger this. however, this affects performance very hard, but stabilizes learning
	let survival = 4; //probably keep this
	let badsurv = 1; //probably keep this
	let prob_avg = 0.1; //probably keep this, does not change very much
	let prob_mut = 0.95; //probably keep this, not too much change
	let prob_op = 0.75; //the bigger the network, the lower this (try op_range first)
	let op_range = 0.25; //the bigger the network, the lower this (lower this before prob_op)
	let prob_block = 0.02; //can be adjusted, so the network does not adds too many layers or adds more layers (no added blocks probably means prob_op or op_range too high)
	let prob_new = 0.1; //can be adjusted, but do not set to 0.0, does not change too much, but avoids getting stuck
	
    // create a new neural network, evaluator and optimizer
	let nn = NN::new(2, 3, 1, Activation::LRELU, Activation::Linear);
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
