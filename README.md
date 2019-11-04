# A Complementary Learning Systems Approach to Temporal Difference Learning

This repo contains the code associated with the paper 'A Complementary Learning Systems Approach to Temporal Difference Learning' (Blakeman and Mareschal, 2020). The paper can be found at the following link: 

https://www.sciencedirect.com/science/article/pii/S0893608019303338?via%3Dihub

## Dependencies

If you are using pip then run the following command in your virtual environment in order to install the required dependencies:

```
pip install -r requirements.txt
```

Otherwise you can manually install the requirements found in ```requirements.txt```

## Grid World Simulations

### Run Simulations

To run an agent on the grid world task simply run the ```RunGridWorld.py``` file:

```
python RunGridWorld.py
```

The function that is called inside this file dictates what simulation is ran. Below is a brief description of the functions available:

```RunRandomSeedSweep()``` - Run an agent on a range of randomly generated grid worlds (Figure 1 in the paper).
```RunMazeTypeSweep()``` - Run an agent on three mazes of increasing difficulty (Figure 4 in the paper)
```RunRevaluationSweep()``` - Run an agent on a simple maze and then introduce an obstacle halfway through training (Figure 5 in the paper)

### Parameters

Parameters for grid world simulations are stored in two dictionaries in ```GridWorld/Parameters.py```

```
maze_params = {'type': MazeType.random,
               'width': 10,
               'height': 10,
               'num_rewards': 1,
               'num_trials': 1000,
               'random_seed': 0,
               'max_steps': 1000,
               'num_repeats': 30
               }

agent_params = {'agent_type': AgentType.CTDL,
                'bSOM': True,
                'SOM_alpha': .01,
                'SOM_sigma': .1,
                'SOM_sigma_const': .1,
                'Q_alpha': .9,
                'w_decay': 10,
                'TD_decay': 1,
                'SOM_size': 6,
                'e_trials': 200
                }
```

Some of the parameters will be overwritten depending on the type of Simulation being ran e.g. the maze type will be changed if running ```RunMazeTypeSweep()``` or ```RunRevaluationSweep()```. A brief description of each of the parameters can be found below:

Maze Parameters:

* type --> (MazeType) The type of maze to be ran
* width --> (int) The width of the grid world
* height --> (int) The height of the grid world
* num_rewards --> (int) The number of positive rewards in the grid world
* num_trials --> (int) The number of trials for learning
* random_seed --> (int) The random seed used to generate the grid worlds
* max_steps --> (int) The maximum number of steps per trial
* num_repeats --> (int) The number of times to repeat the simulation so that the standard deviation can be calculated

Agent Parameters:

* agent_type --> (AgentType) The type of agent
* bSOM --> (bool) Flag for using the SOM (only applies to agents that have a SOM)
* SOM_alpha --> (float) Learning rate for updating the weights of the SOM
* SOM_sigma --> (float) Standard deviation of the SOM neighbourhood function
* SOM_sigma_const --> (float) Constant for denominator in SOM neighbourhood function
* Q_alpha --> (float) Learning rate for updating the Q values of the SOM
* w_decay --> (float) Temperature for calculating eta
* TD_decay --> (float) Temperature for calculating sigma
* SOM_size --> (int) wdith / height of the SOM (square root of total number of units)
* e_trials --> (int) The number of trials to decay epsilon over (for agents that use an epsilon-greedy policy) 

### Results

Results are stored in ```GridWorld/Results/``` and each simulation is saved in a seperate folder. The folder is named according to the date and time the simulation was ran.

### Plotting

All plots will be saved in ```GridWord/Plots/```. To generate plots run either ```GridWorld/AnalyseRandomSeedSweep.py```, ```GridWorld/AnalyseMazeTypeSweep.py``` or ```GridWorld/RevaluationSweep.py``` depending on the simulation that you have run and want to analyse. At the top of each file is a variable called ```dir``` that specifies the subdirectory of the results you want to analyse. For example if you have just ran a random seed sweep then you might move the results into a new subdirectory ```GridWorld/Results/RandomSeedSweep1/``` and then change ```dir = "RandomSeedSweep1"```. In addition to the ```dir``` variable there is also a ```to_compare``` variable that you can use to specify the names of the agents that you want to plot against each other for comparison purposes.

## Gym Simulations (Cart-Pole and Continuous Mountain Car)

### Run Simulations

To run an agent in the OpenAI Gym environment simply run the ```RunGym.py``` file:

```
python RunGridWorld.py
```

### Parameters

Parameters for gym simulations are stored in two dictionaries in ```Gym/Parameters.py```. They are largely the same as those used for the gird world simulations (see above). To switch between the Cart-Pole and Continuous Mountain Car tasks simple change the ```'env'``` parameter using the EnvType enum. Remember to change the corresponding ```'agent_type'``` as well using the AgentType enum e.g. the Continuous Mountain Car task uses A2C and CTDL_A2C as they allow for continuous action spaces.

### Results

Results are stored in ```Gym/Results/``` and each simulation is saved in a seperate folder. The folder is named according to the date and time the simulation was ran.

### Plotting

The instructions for plotting are similar to the grid world simulations (see above). The main difference is you need to run ```Gym/AnalyseResults.py```, which will look in ```Gym/Results/``` for the subdirectory specified by the variable ```dir```.

## Authors

* **Sam Blakeman** - *Corresponding Author* - sam.blakeman.15@ucl.ac.uk
* **Denis Mareschal**

## Acknowledgments

* BBSRC for funding the research
* NVIDIA for providing the GPU used to run the simulations
