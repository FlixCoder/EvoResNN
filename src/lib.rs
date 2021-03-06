//! Modified NN, originally from: https://github.com/jackm321/RustNN
//! Most of the original code is gone.

#[macro_use]
extern crate serde_derive;

extern crate serde;
extern crate serde_json;
extern crate rand;
extern crate rayon;

use std::cmp::Ordering;
use rand::Rng;
use rand::distributions::{Normal, IndependentSample};
use rayon::prelude::*;

//values for a (0,1) distribution (so (-1, 1) interval in standard deviation)
//const SELU_FACTOR_A:f64 = 1.0507; //greater than 1, lambda in https://arxiv.org/pdf/1706.02515.pdf
//const SELU_FACTOR_B:f64 = 1.6733; //alpha in https://arxiv.org/pdf/1706.02515.pdf
//values for a (0,2) distribution (so (-2, 2) interval in standard deviation)
const SELU_FACTOR_A:f64 = 1.06071; //greater than 1, lambda in https://arxiv.org/pdf/1706.02515.pdf
const SELU_FACTOR_B:f64 = 1.97126; //alpha in https://arxiv.org/pdf/1706.02515.pdf

const PELU_FACTOR_A:f64 = 1.5;
const PELU_FACTOR_B:f64 = 2.0;

const LRELU_FACTOR:f64 = 0.33;


/// Specifies the activation function
#[derive(Debug, Copy, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub enum Activation
{
	/// Sigmoid activation
	Sigmoid,
	/// SELU activation
	SELU,
	/// PELU activation
	PELU,
	/// Leaky ReLU activation
	LRELU,
	/// Linear activation
	Linear,
	/// Tanh activation
	Tanh,
	/// Quadratic activation
	Quadratic,
	/// Cubic activation
	Cubic,
	/// Rectified linear activation
	RELU,
	/// Edgy soft Tanh activation (clip linear in [-1,1])
	ESTAN
}

/// Neural network
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct NN
{
	generation: u32, //generation of current network
	blocks: u32, //number of residual layer blocks
	num_inputs: u32, //number of inputs to NN
	hidden_size: u32, //size of every hidden layer. always 2 layers in a block. at least 1 hidden layer
	num_outputs: u32, //number of outputs of the NN
	hid_act: Activation, //hidden layer activation
	out_act: Activation, //output layer activation
	layers: Vec<Vec<Vec<f64>>>, //NN layers -> nodes -> weights
}

impl NN
{
	/// Creates a new neural net with the given parameters. Initially there is one hidden layer
	/// Be careful with Sigmoid as hidden layer activation function, as it could possibly slow down block additions
	pub fn new(inputs:u32, hidden_size:u32, outputs:u32, hidden_activation:Activation, output_activation:Activation) -> NN
	{
		let mut rng = rand::thread_rng();

		if inputs < 1 || hidden_size < 1 || outputs < 1
		{
			panic!("inappropriate parameter bounds");
		}

		// setup the layers
		let mut layers = Vec::new();
		let mut prev_layer_size = inputs;
		for i in 0..2
		{ //one hidden layer and one output layer
			let mut layer: Vec<Vec<f64>> = Vec::new();
			let layer_size = if i == 1 { outputs } else { hidden_size };
			let mut init_std_scale = 2.0; //He init
			if hidden_activation == Activation::SELU { init_std_scale = 1.0; } //MSRA / Xavier init
			let normal = Normal::new(0.0, (init_std_scale / prev_layer_size as f64).sqrt());
			for _ in 0..layer_size
			{
				let mut node: Vec<f64> = Vec::with_capacity(1 + prev_layer_size as usize);
				for i in 0..prev_layer_size+1
				{
					if i == 0 //threshold aka bias
					{
						node.push(0.0);
					}
					else
					{
						let random_weight: f64 = normal.ind_sample(&mut rng);
						node.push(random_weight);
					}
				}
				layer.push(node);
			}
			layer.shrink_to_fit();
			layers.push(layer);
			prev_layer_size = layer_size;
		}
		layers.shrink_to_fit();
		
		//set activation functions
		NN { generation: 0, blocks: 0, num_inputs: inputs, hidden_size: hidden_size, num_outputs: outputs,
				hid_act: hidden_activation, out_act: output_activation, layers: layers }
	}
	
	///Runs a feed-forward run through the network and returns the results (of the output layer)
	pub fn run(&self, inputs:&[f64]) -> Vec<f64>
	{
		if inputs.len() as u32 != self.num_inputs
		{
			panic!("Input has a different length than the network's input layer!");
		}
		
		self.do_run(inputs, 1.0).pop().unwrap()
	}
	
	///Runs a feed-forward run through the network using random drop-out and returns the results (of the output layer)
	///k = probability to keep the node respectively (must be in (0.0, 1.0])
	pub fn run_dropout(&self, inputs:&[f64], k:f64) -> Vec<f64>
	{
		if inputs.len() as u32 != self.num_inputs
		{
			panic!("Input has a different length than the network's input layer!");
		}
		if k <= 0.0 || k > 1.0
		{
			panic!("Probability to keep nodes is not in the correct range! (0.0, 1.0]");
		}
		
		self.do_run(inputs, k).pop().unwrap()
	}

	/// Encodes the network as a JSON string.
	pub fn to_json(&self) -> String
	{
		serde_json::to_string(self).ok().expect("Encoding JSON failed!")
	}

	/// Builds a new network from a JSON string.
	pub fn from_json(encoded:&str) -> NN
	{
		let network:NN = serde_json::from_str(encoded).ok().expect("Decoding JSON failed!");
		network
	}

	fn do_run(&self, inputs:&[f64], k:f64) -> Vec<Vec<f64>>
	{
		let mut rng = rand::thread_rng();
		let mut results = Vec::new();
		results.push(inputs.to_vec());
		let num_layers = self.layers.len();
		for (layer_index, layer) in self.layers.iter().enumerate()
		{
			let mut layer_results = Vec::new();
			for (i, node) in layer.iter().enumerate()
			{
				if k == 1.0 || rng.gen::<f64>() < k
				{ //keep node => calculate results
					let mut sum = modified_dotprod(&node, &results[layer_index]); //sum of forward pass to this node
					//residual network shortcut
					if layer_index >= 1 && layer_index < num_layers - 1 && layer_index % 2 == 0
					{
						sum += results[layer_index - 1][i];
					}
					//standard forward pass activation
					let act;
					if layer_index == self.layers.len()-1 //output layer
					{ act = self.out_act; }
					else { act = self.hid_act; }
					sum = match act {
								Activation::Sigmoid => sigmoid(sum),
								Activation::SELU => selu(sum),
								Activation::PELU => pelu(sum),
								Activation::LRELU => lrelu(sum),
								Activation::Linear => linear(sum),
								Activation::Tanh => tanh(sum),
								Activation::Quadratic => quad(sum),
								Activation::Cubic => cubic(sum),
								Activation::RELU => relu(sum),
								Activation::ESTAN => estan(sum),
							};
					//when using dropout multiply to a factor to keep the variance/norm
					sum /= k; //for d = 0.0 nothing happens
					//push result
					layer_results.push(sum);
				}
				else
				{ //drop node => result 0 so it has no effect
					layer_results.push(0.0);
				}
			}
			results.push(layer_results);
		}
		results
	}

	fn get_layers_mut(&mut self) -> &mut Vec<Vec<Vec<f64>>>
	{
		&mut self.layers
	}

	fn get_layers(&self) -> &Vec<Vec<Vec<f64>>>
	{
		&self.layers
	}

	pub fn get_inputs(&self) -> u32
	{
		self.num_inputs
	}

	pub fn get_hidden(&self) -> u32
	{
		self.hidden_size
	}

	pub fn get_outputs(&self) -> u32
	{
		self.num_outputs
	}

	pub fn get_hid_act(&self) -> Activation
	{
		self.hid_act
	}

	pub fn get_out_act(&self) -> Activation
	{
		self.out_act
	}

	fn set_hid_act(&mut self, act:Activation)
	{
		self.hid_act = act;
	}

	fn set_out_act(&mut self, act:Activation)
	{
		self.out_act = act;
	}

	pub fn get_gen(&self) -> u32
	{
		self.generation
	}

	fn set_gen(&mut self, gen:u32)
	{
		self.generation = gen;
	}

	pub fn get_blocks(&self) -> u32
	{
		self.blocks
	}

	///  breed a child from the 2 networks, either by random select or by averaging weights
	/// panics if the neural net's hidden_size are not the same
	pub fn breed(&self, other:&NN, prob_avg:f64) -> NN
	{
		let mut rng = rand::thread_rng();
		let mut newnn = self.clone();
		
		//set generation
		let oldgen = newnn.get_gen();
		newnn.set_gen((other.get_gen() + oldgen + 3) / 2); //round up and + 1
		
		//set activation functions
		if rng.gen::<f64>() < 0.5
		{ //else is already set to own activation
			newnn.set_hid_act(other.get_hid_act());
		}
		if rng.gen::<f64>() < 0.5
		{ //else is already set to own activation
			newnn.set_out_act(other.get_out_act());
		}
		
		//set parameters
		{ //put in scope, because of mutable borrow before ownership return
			let layers1 = newnn.get_layers_mut();
			let layers1len = layers1.len();
			let layers2 = other.get_layers();
			for layer_index in 0..layers1len
			{
				let layer = &mut layers1[layer_index];
				for node_index in 0..layer.len()
				{
					let node = &mut layer[node_index];
					for weight_index in 0..node.len()
					{
						let mut layer2val = 0.0;
						if layer_index == layers1len - 1 //last layer
						{ //use the same layer weights again for the output layer, also if network 2 is deeper
							let outlayer_i = layers2.len() - 1;
							layer2val = layers2[outlayer_i][node_index][weight_index];
						}
						else if layer_index < layers2.len() - 1
						{ //simulate same network size by using zeros for the block (so identity if not sigmoid)
							layer2val = layers2[layer_index][node_index][weight_index];
						} //if layers2 is deeper than layers1, the shorter layers1 is taken and deeper layers ignored
						
						if prob_avg == 1.0 || (prob_avg != 0.0 && rng.gen::<f64>() < prob_avg)
						{ //average between weights
							node[weight_index] = (node[weight_index] + layer2val) / 2.0;
						}
						else
						{
							if rng.gen::<f64>() < 0.5
							{ //random if stay at current weight or take father's/mother's
								node[weight_index] = layer2val;
							}
						}
					}
				}
			}
		}
		
		//return
		newnn
	}

	/// mutate the current network
	/// params: (all probabilities in [0,1])
	/// prob_new:f64 - probability to become a new freshly initialized network of same size/architecture (to change hidden size create one manually and don't breed them)
	/// prob_block:f64 - probability to add another residual block (2 layers) somewhere in the network, initially close to identity if not sigmoid (double activation), random prob_op afterwards
	/// prob_op:f64 - probability to apply an operation to a weight
	/// op_range:f64 - maximum positive or negative adjustment of a weight
	// ideas to add:
	// 			change activation function or at least activation function parameters
	//			remove blocks (weight to zero does not work)
	//			still use backprop for something to speed up calculation
	pub fn mutate(&mut self, prob_new:f64, prob_block:f64, prob_op:f64, op_range:f64)
	{
		let mut rng = rand::thread_rng();
		//fresh random network parameters
		if rng.gen::<f64>() < prob_new
		{
			self.mutate_new();
		}
		//random residual block addition
		if rng.gen::<f64>() < prob_block
		{
			self.mutate_block();
		}
		//random addition / substraction op mutation
		if prob_op != 0.0 && op_range != 0.0
		{
			self.mutate_op(prob_op, op_range);
		}
	}

	/// adds an additional residual block somewhere
	fn mutate_block(&mut self)
	{
		let mut rng = rand::thread_rng();
		let mut place:usize = 1; //index of layer where to put the new block
		if self.blocks != 0
		{
			place += 2 * (rng.gen::<usize>() % (1 + self.blocks as usize));
		}
		
		//insert block
		let mut layer1:Vec<Vec<f64>> = Vec::with_capacity(self.hidden_size as usize);
		let mut layer2:Vec<Vec<f64>> = Vec::with_capacity(self.hidden_size as usize);
		for _ in 0..self.hidden_size
		{
			layer1.push(vec![0.0; 1 + self.hidden_size as usize]);
			layer2.push(vec![0.0; 1 + self.hidden_size as usize]);
		}
		self.layers.insert(place, layer2);
		self.layers.insert(place, layer1);
		
		self.blocks += 1;
	}

	fn mutate_new(&mut self)
	{
		let mut rng = rand::thread_rng();
		let mut init_std_scale = 2.0; //He init
		if self.hid_act == Activation::SELU { init_std_scale = 1.0; } //MSRA / Xavier init
		let mut prev_layer_size = self.num_inputs as usize;
		for layer_index in 0..self.layers.len()
		{
			let layer = &mut self.layers[layer_index];
			let normal = Normal::new(0.0, (init_std_scale / prev_layer_size as f64).sqrt());
			for node_index in 0..layer.len()
			{
				let node = &mut layer[node_index];
				for weight_index in 0..node.len()
				{
					node[weight_index] = if weight_index == 0 { 0.0 } else { normal.ind_sample(&mut rng) };
				}
			}
			prev_layer_size = layer.len();
		}
	}

	/// mutate using addition/substraction of a random value (random per node)
	fn mutate_op(&mut self, prob_op:f64, op_range:f64)
	{
		let mut rng = rand::thread_rng();
		for layer_index in 0..self.layers.len()
		{
			let layer = &mut self.layers[layer_index];
			for node_index in 0..layer.len()
			{
				let node = &mut layer[node_index];
				for weight_index in 0..node.len()
				{
					//possibly modify weight
					if rng.gen::<f64>() < prob_op
					{ //RNG says this weight will be changed
						let delta = op_range * (2.0 * rng.gen::<f64>() - 1.0);
						node[weight_index] += delta;
					}
				}
			}
		}
	}
}
	
fn sigmoid(x:f64) -> f64
{
    1f64 / (1f64 + (-x).exp())
}

fn selu(x:f64) -> f64
{ //SELU activation
	SELU_FACTOR_A * if x < 0.0
	{
		SELU_FACTOR_B * x.exp() - SELU_FACTOR_B
	}
	else
	{
		x
	}
}

fn pelu(x:f64) -> f64
{ //PELU activation
	if x < 0.0
	{
		PELU_FACTOR_A * (x / PELU_FACTOR_B).exp() - PELU_FACTOR_A
	}
	else
	{
		(PELU_FACTOR_A / PELU_FACTOR_B) * x
	}
}

fn lrelu(x:f64) -> f64
{ //LRELU activation
	if x < 0.0
	{
		LRELU_FACTOR * x
	}
	else
	{
		x
	}
}

fn linear(x:f64) -> f64
{ //linear activation
	x
}

fn tanh(x:f64) -> f64
{ //tanh activation
	x.tanh()
}

fn quad(x:f64) -> f64
{
	x * x
}

fn cubic(x:f64) -> f64
{
	x * x * x
}

fn relu(x:f64) -> f64
{
	x.max(0.0)
}

fn estan(x:f64) -> f64
{
	x.min(1.0).max(-1.0)
}

fn modified_dotprod(node: &Vec<f64>, values: &Vec<f64>) -> f64
{
    let mut it = node.iter();
    let mut total = *it.next().unwrap(); //start with the threshold weight
    for (weight, value) in it.zip(values.iter())
	{
        total += weight * value;
    }
    total
}



/// Trait to define evaluators in order to use the algorithm in a flexible way
pub trait Evaluator
{
	fn evaluate(&self, nn:&NN) -> f64; //returns rating of NN (higher is better (you can inverse with -))
}

/// Optimizer class to optimize neural nets by evolutionary / genetic algorithms
/// For parallel optimization the evaluator has to implement the Sync-trait! Regarding controlling the number of threads, see Rayon's documentation
pub struct Optimizer<T:Evaluator>
{
	eval: T, //evaluator
	nets: Vec<(NN, f64)>, //population of nets and ratings (sorted, high/best rating in front)
}

impl<T:Evaluator> Optimizer<T>
{
	/// Create a new optimizer using the given evaluator for the given neural net
	pub fn new(evaluator:T, nn:NN) -> Optimizer<T>
	{
		let mut netvec = Vec::new();
		let rating = evaluator.evaluate(&nn);
		netvec.push((nn, rating));
		
		Optimizer { eval: evaluator, nets: netvec }
	}
	
	/// Get a reference to the population
	pub fn get_population(&self) -> &Vec<(NN, f64)>
	{
		&self.nets
	}
	
	/// Get a mutable reference to the population (there is no set_population, use this)
	pub fn get_population_mut(&mut self) -> &mut Vec<(NN, f64)>
	{
		&mut self.nets
	}
	
	/// Save population as json string and return it
	pub fn save_population(&self) -> String
	{
		serde_json::to_string(&self.nets).ok().expect("Encoding JSON failed!")
	}
	
	/// Load population from json string
	pub fn load_population(&mut self, encoded:&str)
	{
		self.nets = serde_json::from_str(encoded).ok().expect("Decoding JSON failed!");
	}
	
	/// Get a reference to the used evaluator
	pub fn get_eval(&self) -> &T
	{
		&self.eval
	}
	
	/// Switch to a new evaluator to allow change of evaluation. you should probably call reevaluate afterwards
	pub fn set_eval(&mut self, evaluator:T)
	{
		self.eval = evaluator;
	}
	
	/// Clones the best NN an returns it
	pub fn get_nn(&mut self) -> NN
	{
		self.nets[0].0.clone()
	}
	
	/// Reevaluates all neural nets based on the current (possibly changed) evaluator, returns best score
	pub fn reevaluate(&mut self) -> f64
	{
		//evaluate
		for net in &mut self.nets
		{
			net.1 = self.eval.evaluate(&net.0);
		}
		//sort and return
		self.sort_nets();
		self.nets[0].1
	}
	
	/// Same as reevaluate but in parallel
	pub fn reevaluate_par(&mut self) -> f64
		where T:Sync
	{
		//evaluate
		{ //scope for borrowing
			let eval = &self.eval;
			self.nets.par_iter_mut().for_each(|net|
				{
					net.1 = eval.evaluate(&net.0);
				});
		}
		//sort and return
		self.sort_nets();
		self.nets[0].1
	}
	
	/// Optimizes the NN for the given number of generations.
	/// Returns the rating of the best NN score afterwards.
	/// 
	/// Parameters: (probabilities are in [0,1]) (see xor example for help regarding paramter choice)
	/// generations - number of generations to optimize over
	/// population - size of population to grow up to
	/// survival - number nets to survive by best rating
	/// bad_survival - number of nets to survive randomly from nets, that are not already selected to survive from best rating
	/// prob_avg - probability to use average weight instead of selection in breeding
	/// prob_mut - probability to mutate after breed
	/// prob_new - probability to generate a new random network
	/// prob_block - probability to add another residual block
	/// prob_op - probability for each weight to mutate using an delta math operation during mutation
	/// op_range - factor to control the range in which delta can be in
	pub fn optimize(&mut self, generations:u32, population:u32, survival:u32, bad_survival:u32, prob_avg:f64, prob_mut:f64,
					prob_new:f64, prob_block:f64, prob_op:f64, op_range:f64) -> f64
	{
		//optimize for "generations" generations
		for _ in 0..generations
		{
			let children = self.populate(population as usize, prob_avg, prob_mut, prob_new, prob_block, prob_op, op_range);
			self.evaluate(children);
			self.sort_nets();
			self.survive(survival, bad_survival);
			//self.sort_nets(); //not needed, because population generation is choosing randomly
		}
		//return best rating
		self.sort_nets();
		self.nets[0].1
	}
	
	/// Same as optimize, but in parallel.
	pub fn optimize_par(&mut self, generations:u32, population:u32, survival:u32, bad_survival:u32, prob_avg:f64, prob_mut:f64,
					prob_new:f64, prob_block:f64, prob_op:f64, op_range:f64) -> f64
		where T:Sync
	{
		//optimize for "generations" generations
		for _ in 0..generations
		{
			let children = self.populate(population as usize, prob_avg, prob_mut, prob_new, prob_block, prob_op, op_range);
			self.evaluate_par(children);
			self.sort_nets();
			self.survive(survival, bad_survival);
			//self.sort_nets(); //not needed, because population generation is choosing randomly!
		}
		//return best rating
		self.sort_nets();
		self.nets[0].1
	}
	
	/// Easy shortcut to optimize using standard parameters.
	/// For paramters see optimize.
	pub fn optimize_easy(&mut self, generations:u32, population:u32, prob_block:f64, prob_op:f64, op_range:f64) -> f64
	{
		//standard parameter choice
		let survival = 4;
		let badsurv = 1;
		let prob_avg = 0.1;
		let prob_mut = 0.95;
		let prob_new = 0.1;
		self.optimize(generations, population, survival, badsurv, prob_avg, prob_mut, prob_new, prob_block, prob_op, op_range)
	}
	
	/// Same as optimize_easy, but in parallel
	pub fn optimize_easy_par(&mut self, generations:u32, population:u32, prob_block:f64, prob_op:f64, op_range:f64) -> f64
		where T:Sync
	{
		//standard parameter choice
		let survival = 4;
		let badsurv = 1;
		let prob_avg = 0.1;
		let prob_mut = 0.95;
		let prob_new = 0.1;
		self.optimize_par(generations, population, survival, badsurv, prob_avg, prob_mut, prob_new, prob_block, prob_op, op_range)
	}
	
	/// Generate initial random population of the given size.
	/// Just a shortcut to optimize with less parameters.
	pub fn gen_population(&mut self, population:u32) -> f64
	{
		self.optimize(1, population, population, 0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
	}
	
	/// Generates new population and returns a vec of nets, that need to be evaluated
	fn populate(&self, size:usize, prob_avg:f64, prob_mut:f64, prob_new:f64, prob_block:f64, prob_op:f64, op_range:f64) -> Vec<(NN, f64)>
	{
		let mut rng = rand::thread_rng();
		let len = self.nets.len();
		let missing = size - len;
		let mut newpop = Vec::new();
		
		for _ in 0..missing
		{
			let i1:usize = rng.gen::<usize>() % len;
			let i2:usize = rng.gen::<usize>() % len;
			let othernn = &self.nets[i2].0;
			let mut newnn = self.nets[i1].0.breed(othernn, prob_avg);
			
			if rng.gen::<f64>() < prob_mut
			{
				newnn.mutate(prob_new, prob_block, prob_op, op_range);
			}
			
			newpop.push((newnn, 0.0));
		}
		
		newpop
	}
	
	/// Evaluates a given set of NNs and appends them into the internal storage
	fn evaluate(&mut self, mut nets:Vec<(NN, f64)>)
	{
		for net in &mut nets
		{
			net.1 = self.eval.evaluate(&net.0);
		}
		self.nets.append(&mut nets);
	}
	
	/// Same as evaluate but in parallel
	fn evaluate_par(&mut self, mut nets:Vec<(NN, f64)>)
		where T:Sync
	{
		nets.par_iter_mut().for_each(|net|
			{
				net.1 = self.eval.evaluate(&net.0);
			});
		self.nets.append(&mut nets);
	}
	
	/// Eliminates population, so that the best "survival" nets and random "bad_survival" nets survive
	fn survive(&mut self, survival:u32, bad_survival:u32)
	{
		if survival as usize >= self.nets.len() { return; } //already done
		
		let mut rng = rand::thread_rng();
		let mut bad = self.nets.split_off(survival as usize);
		
		for _ in 0..bad_survival
		{
			if bad.is_empty() { return; }
			let i:usize = rng.gen::<usize>() % bad.len();
			self.nets.push(bad.swap_remove(i));
		}
	}
	
	/// Sorts the internal NNs, so that the best net is in front (index 0)
	fn sort_nets(&mut self)
	{ //best nets (high score) in front, bad and NaN nets at the end
		self.nets.sort_by(|ref r1, ref r2| { //reverse partial cmp and check for NaN
				let r = (r2.1).partial_cmp(&r1.1);
				if r.is_some() { r.unwrap() }
				else
				{
					if r1.1.is_nan() { if r2.1.is_nan() { Ordering::Equal } else { Ordering::Greater } } else { Ordering::Less }
				}
			});
	}
}
