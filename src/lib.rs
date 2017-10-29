//! Modified NN, originally from: https://github.com/jackm321/RustNN
//! Most of the original code is gone.

#[macro_use]
extern crate serde_derive;

extern crate serde;
extern crate serde_json;
extern crate rand;

use std::cmp::Ordering;
use self::rand::Rng;
use self::rand::distributions::{Normal, IndependentSample};

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
#[derive(Debug, Copy, Clone, PartialEq)]
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
}

/// Neural network
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NN
{
	generation: u32, //generation of current network
	blocks: u32, //number of residual layer blocks
	num_inputs: u32, //number of inputs to NN
	hidden_size: u32, //size of every hidden layer. always 2 layers in a block. at least 1 hidden layer
	hid_act: u32, //hidden layer activation
	out_act: u32, //output layer activation
	layers: Vec<Vec<Vec<f64>>>, //NN layers -> nodes -> weights
}

impl NN
{
	/// Creates a new neural net with the given parameters. Initially there is one hidden layer
	/// Be careful with Sigmoid as hidden layer activation function, as sigmoid(0) != 0, which could
	/// possibly cause the algorithm to drop neural nets with an added layer, because it does not represent the identity
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
                layer.push(node)
            }
            layer.shrink_to_fit();
            layers.push(layer);
            prev_layer_size = layer_size;
        }
        layers.shrink_to_fit();
		
		//set activation functions
		let hid_act = match hidden_activation {
			Activation::Sigmoid => 0,
			Activation::SELU => 1,
			Activation::PELU => 2,
			Activation::LRELU => 3,
			Activation::Linear => 4,
			Activation::Tanh => 5,
			Activation::Quadratic => 6,
			Activation::Cubic => 7,
		};
		let out_act = match output_activation {
			Activation::Sigmoid => 0,
			Activation::SELU => 1,
			Activation::PELU => 2,
			Activation::LRELU => 3,
			Activation::Linear => 4,
			Activation::Tanh => 5,
			Activation::Quadratic => 6,
			Activation::Cubic => 7,
		};
        NN { layers: layers, num_inputs: inputs, hidden_size: hidden_size, hid_act: hid_act, out_act: out_act, blocks: 0, generation: 0 }
    }
	
    pub fn run(&self, inputs: &[f64]) -> Vec<f64>
	{
        if inputs.len() as u32 != self.num_inputs
		{
            panic!("input has a different length than the network's input layer");
        }
        self.do_run(inputs).pop().unwrap()
    }
	
	/// Encodes the network as a JSON string.
    pub fn to_json(&self) -> String
	{
        serde_json::to_string(self).ok().expect("encoding JSON failed!")
    }

	/// Builds a new network from a JSON string.
    pub fn from_json(encoded: &str) -> NN
	{
        let network: NN = serde_json::from_str(encoded).ok().expect("Decoding JSON failed!");
        network
    }
	
    fn do_run(&self, inputs: &[f64]) -> Vec<Vec<f64>>
	{
        let mut results = Vec::new();
        results.push(inputs.to_vec());
		let num_layers = self.layers.len();
        for (layer_index, layer) in self.layers.iter().enumerate()
		{
            let mut layer_results = Vec::new();
            for (i, node) in layer.iter().enumerate()
			{
				let mut sum = modified_dotprod(&node, &results[layer_index]); //sum of forward pass to this node
				//residual network shortcut
				if layer_index >= 1 && layer_index < num_layers - 1 && layer_index % 2 == 0
				{
					sum += results[layer_index - 1][i];
				}
				//standard forward pass activation
				layer_results.push( if layer_index == self.layers.len()-1 //output layer
					{
						match self.out_act {
							0 => sigmoid(sum), //sigmoid
							1 => selu(sum), //selu
							2 => pelu(sum), //pelu
							3 => lrelu(sum), //lrelu
							4 => linear(sum), //linear
							5 => tanh(sum), //tanh
							6 => quad(sum), //quadratic
							_ => cubic(sum), //cubic
						}
					}
					else
					{
						match self.hid_act {
							0 => sigmoid(sum), //sigmoid
							1 => selu(sum), //selu
							2 => pelu(sum), //pelu
							3 => lrelu(sum), //lrelu
							4 => linear(sum), //linear
							5 => tanh(sum), //tanh
							6 => quad(sum), //quadratic
							_ => cubic(sum), //cubic
						}
					} );
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
	
	fn get_hid_act(&self) -> u32
	{
		self.hid_act
	}
	
	fn get_out_act(&self) -> u32
	{
		self.out_act
	}
	
	fn set_hid_act(&mut self, act:u32)
	{
		self.hid_act = act;
	}
	
	fn set_out_act(&mut self, act:u32)
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
			let mut layers1 = newnn.get_layers_mut();
			let layers1len = layers1.len();
			let layers2 = other.get_layers();
			for layer_index in 0..layers1len
			{
				let mut layer = &mut layers1[layer_index];
				for node_index in 0..layer.len()
				{
					let mut node = &mut layer[node_index];
					for weight_index in 0..node.len()
					{
						let mut layer2val = 0.0;
						if layer_index < layers2.len() - 1
						{ //simulate same network size by using zeros for the block
							layer2val = layers2[layer_index][node_index][weight_index];
						} //if layers2 is deeper than layers1, the shorter layers1 is taken and deeper layers ignored
						else if layer_index == layers1len - 1 //last layer
						{ //use the same layer weights again for the output layer
							let outlayer_i = layers2.len() - 1;
							layer2val = layers2[outlayer_i][node_index][weight_index];
						}
						
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
	/// prob_op:f64 - probability to apply an addition/substraction to a node
	/// op_range:f64 - maximum positive or negative adjustment of a weight
	/// prob_block:f64 - probability to add another residual block (2 layers) somewhere in the network, initially identity, random prob_op afterwards
	/// prob_new:f64 - probability to become a new freshly initialized network of same size/architecture (to change hidden size create one manually and don't breed them)
	// ideas to add:
	// 			change activation function or at least activation function parameters,
	//			still use backprop for something to speed up calculation
	//			zero out some nodes or blocks or remove things
	pub fn mutate(&mut self, prob_op:f64, op_range:f64, prob_block:f64, prob_new:f64)
	{
		let mut rng = rand::thread_rng();
		self.increment_generation();
		//random residual block addition
		if rng.gen::<f64>() < prob_block
		{
			self.mutate_block();
		}
		//fresh random network parameters
		if rng.gen::<f64>() < prob_new
		{
			self.generation /= 2; //as the blocks stay the same, don't set to 0, but decrease significantly
			self.mutate_new();
		}
		//random addition / substraction op mutation
		if prob_op != 0.0 && op_range != 0.0
		{
			self.mutate_op(prob_op, op_range);
		}
	}
	
	fn increment_generation(&mut self)
	{
		self.generation += 1;
	}
	
	/// adds an additional residual block somewhere, representing the identity function (unless sigmoid)
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
		if self.hid_act == 1 { init_std_scale = 1.0; } //MSRA / Xavier init
		let mut prev_layer_size = self.num_inputs as usize;
		for layer_index in 0..self.layers.len()
		{
			let mut layer = &mut self.layers[layer_index];
			let normal = Normal::new(0.0, (init_std_scale / prev_layer_size as f64).sqrt());
			for node_index in 0..layer.len()
			{
				let mut node = &mut layer[node_index];
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
            let mut layer = &mut self.layers[layer_index];
            for node_index in 0..layer.len()
			{
                let mut node = &mut layer[node_index];
                for weight_index in 0..node.len()
				{
                    let mut delta = 0.0;
					if rng.gen::<f64>() < prob_op
					{
						delta = op_range * (2.0 * rng.gen::<f64>() - 1.0);
					}
                    node[weight_index] += delta;
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

fn modified_dotprod(node: &Vec<f64>, values: &Vec<f64>) -> f64
{
    let mut it = node.iter();
    let mut total = *it.next().unwrap(); // start with the threshold weight
    for (weight, value) in it.zip(values.iter())
	{
        total += weight * value;
    }
    total
}



/// trait to define evaluators in order to use the algorithm in a flexible way
pub trait Evaluator
{
	fn evaluate(&mut self, nn:&NN) -> f64; //returns rating of NN (higher is better (you can inverse with -))
}

/// Optimizer class to optimize neural nets by evolutionary / genetic algorithms
pub struct Optimizer
{
	eval: Box<Evaluator>, //evaluator
	nets: Vec<(NN, f64)>, //population of nets and ratings (sorted, high/best rating in front)
}

impl Optimizer
{
	/// create a new optimizer using the given evaluator for the given neural net
	pub fn new(mut evaluator:Box<Evaluator>, nn:NN) -> Optimizer
	{
		let mut netvec = Vec::new();
		let rating = evaluator.as_mut().evaluate(&nn);
		netvec.push((nn, rating));
		
		Optimizer { eval: evaluator, nets: netvec }
	}
	
	/// returns a mutable borrow of the evaluator to allow adjustments
	pub fn get_eval(&mut self) -> &mut Evaluator
	{
		self.eval.as_mut()
	}
	
	/// optimize the NN for the given number of generations
	/// it is recommended to run a single generation with prob_mut = 1.0 and prob_new = 1.0 at the start to generate the starting population
	/// returns the rating of the best NN afterwards
	/// 
	/// parameters: (probabilities are in [0,1])
	/// generations - number of generations to optimize over
	/// population - size of population to grow up to
	/// survival - number nets to survive by best rating
	/// bad_survival - number of nets to survive randomly from nets, that are not already selected to survive from best rating
	/// prob_avg - probability to use average weight instead of selection in breeding
	/// prob_mut - probability to mutate after breed
	/// prob_op - probability for each weight to mutate using an delta math operation during mutation
	/// op_range - factor to control the range in which delta can be in
	/// prob_block - probability to add another residual block
	/// prob_new - probability to generate a new random network
	pub fn optimize(&mut self, generations:u32, population:u32, survival:u32, bad_survival:u32, prob_avg:f64, prob_mut:f64, prob_op:f64, op_range:f64, prob_block:f64, prob_new:f64) -> f64
	{
		//optimize for generations generations
		for _ in 0..generations
		{
			let children = self.populate(population as usize, prob_avg, prob_mut, prob_op, op_range, prob_block, prob_new);
			self.evaluate(children);
			self.sort_nets();
			self.survive(survival, bad_survival);
			//self.sort_nets(); //not needed, because population generation is choosing randomly
		}
		//return best rating
		self.sort_nets();
		self.nets[0].1
	}
	
	/// generates new population and returns a vec of nets, that need to be evaluated
	fn populate(&self, size:usize, prob_avg:f64, prob_mut:f64, prob_op:f64, op_range:f64, prob_block:f64, prob_new:f64) -> Vec<NN>
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
				newnn.mutate(prob_op, op_range, prob_block, prob_new);
			}
			
			newpop.push(newnn);
		}
		
		newpop
	}
	
	fn evaluate(&mut self, nets:Vec<NN>)
	{
		for nn in nets
		{
			let score = self.get_eval().evaluate(&nn);
			self.nets.push((nn, score));
		}
	}
	
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
	
	/// clones the best NN an returns it
	pub fn get_nn(&mut self) -> NN
	{
		self.nets[0].0.clone()
	}
	
	fn sort_nets(&mut self)
	{ //best nets (high score) in front, bad and NaN nets at the end
		self.nets.sort_by(|ref r1, ref r2| {
				let r = (r2.1).partial_cmp((&r1.1));
				if r.is_some() { r.unwrap() }
				else
				{
					if r1.1.is_nan() { if r2.1.is_nan() {Ordering::Equal} else {Ordering::Greater} } else { Ordering::Less }
				}
			});
	}
}
