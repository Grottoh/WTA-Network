Networks can be trained and tested using <main.py>:
	- Ensure data is present in (by default) <[00LF] data/]>
	- Ensure the results folder exists (by default) <[01LF] results>
	- Uncomment one of lines 266-271 to decide whether to train or test a particular kind of model.
	- When training or testing a model, one can alter hyperparameters in the <argv> variables.
	- When testing a model one needs to provide a path to a trained model.
	
	- Out of lines 266-271 simply uncomment (only) line 266 (<task = "train-hierarchical">) to train a hierarchical network.
	- Out of lines 266-271 simply uncomment (only) line 268 (<task = "train-integration">) to train an integration network.
	
	- Experiment 1 is run by:
	--- Uncommenting line 271 (<task = "train-and-test-integration">)
	--- Having line 719 be <params = param_manager.get_params_00(segment=-1, n_experiments=10)>
	--- BEWARE: this will train and test 2x10=20 integration networks at once, this may be doable on a server but likely not on e.g. your home pc (load can be reduced e.g. by setting <n_experiments=1>)
	
	- Experiment 2.1 is run by (in addition to running experiment 1):
	--- Uncommenting line 271 (<task = "train-and-test-integration">)
	--- Having line 719 be <params = param_manager.get_params_01(segment=-1, n_experiments=10)>
	--- BEWARE: this will train and test 6x10=60 integration networks at once, this may be doable on a server but likely not on e.g. your home pc (load can be reduced e.g. by setting <n_experiments=1>)
	
	- Experiment 2.2 is run by (in addition to running experiment 2.1):
	--- Commenting line 271 (<#task = "train-and-test-integration">) and uncommenting line 270 (<task = "train-and-test-hierarchical">)
	--- Having line 597 be <params = param_manager.get_params_01(segment=-1, n_experiments=10)>
	--- BEWARE: this will train and test 7x10=70 hierarchical networks at once, this may be doable on a server but likely not on e.g. your home pc (load can be reduced e.g. by setting <n_experiments=1>)
	
	- Results will be saved to folders <run-hierarchical_[0-9]{4}> or <run-integration_[0-9]{4}>, which are placed by default in folder <[01LF] results>.
	
	- Complete training and testing of an integration network can take roughly between 1 and 2 days.
	--- This depends of course the speed of your CPU; there is no option to use a GPU.
	--- It also depends largely on the parameters used, but I am assuming default parameters.
	--- Hierarchical networks of course train a lot faster.
	--- No doubt optimization/parallelization could improve these speeds, and in particular a neuromorphic chip would handle this a lot better.
	--- The size of the data is also somewhat larger than is necessary, but this can fairly easily be fixed in future work (or even better, spike-data can just be generated in real-time)