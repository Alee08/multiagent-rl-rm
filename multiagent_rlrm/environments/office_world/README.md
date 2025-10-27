# Experiments - Office World

This project explores multi-agent reinforcement learning using various algorithms in a simulated office environment. The agents interact with the environment based on a set of predefined maps and experiments, aiming to accomplish tasks defined by reward machines.

## Prerequisites

Before running the experiments, ensure you have the following installed:

- Python 3.8 or later
- Required Python packages listed in `requirements.txt`

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Available Configurations
Maps
The following maps are available for the experiments:

- `map1`: A small grid with basic obstacles and goals.
- `map2`: A medium-sized grid with additional complexity.
- `map3`: A large and complex grid with more goals and obstacles.

Stochastic effects can be enabled with ``--stochastic`` argument

## Experiments
Each map can be used with different experiments to test various scenarios:

- `exp1`: Coffee to Office
- `exp2`: E-mail to Office
- `exp3`: Coffee + Email to Office
- `exp4`: A-B-C-D (Patrol) 
- `exp5`: A-B-C-D (Patrol), then Coffee + Email to Office
- `exp6`: A-B-C-D-E (Patrol), then Coffee + Email to Office

## Algorithms
The following algorithms can be used in the experiments:

- `QRMAX`: QR-MAX is a model-based RL algorithm that enhances sample efficiency in NMRDPs by decoupling environment dynamics from reward structures.
- `QRM`: Q-Learning with Reward Machines.
- `QL`: Q-Learning (Cross-Product).
- `RMAX`: A model-based algorithm with optimistic initialization.


## Running Experiments
You can run experiments by executing the office_main.py script. You can specify the map, experiment, and algorithm using command-line arguments. If not specified, the script uses default values from the configuration.

## Usage
```bash
python office_main.py [--map MAP] [--experiment EXPERIMENT] [--algorithm ALGORITHM] [--stochastic] 
                      [--runs RUNS] [--seed SEED] [--save SAVE_PATH] [--load LOAD_PATH] 
                      [--play [EPISODES]] [--render]
```

## Examples
- Run with defaults:
This will use the default map, experiment, and algorithm specified in the `config_office.py`.

```bash
python office_main.py
```
- Run a specific configuration:
This runs the experiment on `map1` with `exp1` using the `QL` algorithm.

```bash
python office_main.py --map map1 --experiment exp1 --algorithm QL
```

- Run another configuration:
This runs `map2` with `exp2` with stochastic action effects using the `QL` algorithm.

```bash
python office_main.py --stochastic --map map2 --experiment exp3 --algorithm QL 
```

- Run another configuration:
This runs `map2` with `exp3` using the `QL` algorithm.

- Run with custom seed and save the policy:
This runs the experiment on map1 with exp1 using the `QRMAX` algorithm and saves the policy to a file.

```bash
python office_main.py --map map1 --experiment exp1 --algorithm QRMAX --runs 0 --seed 3582 --save QRMAX_exp1_policy_3582.pkl
```
- Load a saved policy and execute it:
This loads a pre-trained policy from `QRMAX_exp1_policy_3582.pkl` and executes it for 10 episodes.

```bash
python office_main.py --map map1 --experiment exp1 --algorithm QRMAX --runs 0 --seed 3582 --load policy_1.pkl --play 10
```

## Parameters
- `--stochastic`: Enables stochastic action effects (default: `False`)
- `--map`: Specifies which map to use. Options: `map1`, `map2`, `map3`.
- `--experiment`: Specifies which experiment to run. Options: `exp1`, `exp2`, `exp3`, `exp4`, `exp5`, `exp6`.
- `--algorithm`: Specifies which learning algorithm to use. Options: `QRMAX`, `QRM`, `QL`, `RMAX`.
- `--runs`: Specifies which run(s) to execute. Accepts an integer, list, or range.
- `--seed`: Specifies a custom seed for reproducibility.
- `--save`: Path to save the trained policy.
- `--load`: Path to load a pre-trained policy.
- `--play`: Execute the loaded policy for a specified number of episodes (default is 100).
- `--render`: Enables rendering of the environment.

## Code Structure
- `config.py`: Defines maps, experiments, and agent actions.
- `office_main.py:` Main script to run experiments.
- `ma_office.py`: Office World Environment
- `StateEncoderOfficeWorld`**: Encodes the agent's state, combining its position with the Reward Machine state for use in RL algorithms.
- `PositionEventDetector`**: Detects events based on the agent's current position relative to predefined relevant positions in the environment.

## Results
Results are logged using Weights & Biases (wandb) during training. Metrics such as success rates, rewards, and other performance indicators are tracked per episode, with a focus on evaluating the optimal policy.
If the `--save` option is used, the trained policy is saved to the specified path.
This allows for the policy to be reloaded and tested or executed later using the `--load` and `--play` options.

- Success Rate per Agent:
    The percentage of episodes where each agent successfully completed its task.

- Average Reward per Agent:
    The mean reward obtained by each agent across all test episodes.

- Average Rewards per Step (ARPS) per Agent:
    The average reward per timestep for each agent, offering insight into learning efficiency.

- Average Timesteps:
    The average number of steps required to complete a task across all episodes.



