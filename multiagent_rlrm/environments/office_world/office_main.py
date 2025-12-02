import time, inspect
from multiagent_rlrm.multi_agent.reward_machine import RewardMachine
from multiagent_rlrm.multi_agent.agent_rl import AgentRL
from multiagent_rlrm.learning_algorithms.qlearning import QLearning
from multiagent_rlrm.learning_algorithms.rmax import RMax
from multiagent_rlrm.learning_algorithms.qrmax_v2 import QRMax_v2
from multiagent_rlrm.learning_algorithms.ucbvi import UCBVI
from multiagent_rlrm.learning_algorithms.opsrl import OPSRL
from multiagent_rlrm.render.render import EnvironmentRenderer
from multiagent_rlrm.environments.office_world.state_encoder_office import (
    StateEncoderOfficeWorld,
)
from multiagent_rlrm.environments.office_world.action_encoder_office_world import (
    ActionEncoderOfficeWorld,
)
from multiagent_rlrm.environments.office_world.ma_office import (
    MultiAgentOfficeWorld,
)
from multiagent_rlrm.render.heatmap import (
    generate_heatmaps_for_agents,
    extract_q_tables,
    generate_value_policy_heatmap,
    generate_value_heatmap,
)
from multiagent_rlrm.environments.utils_envs.evaluation_metrics import *
import wandb

from multiagent_rlrm.environments.frozen_lake.detect_event import (
    PositionEventDetector,
)  # Import del nuovo EventDetector
from multiagent_rlrm.multi_agent.wrappers.rm_environment_wrapper import (
    RMEnvironmentWrapper,
)  # Import del wrapper
from config_office import config, get_experiment_for_map
from multiagent_rlrm.utils.utils import *
import logging
import argparse
import signal
import pickle
import matplotlib

matplotlib.use("Agg")  # Imposta il backend "headless"
import matplotlib.pyplot as plt
from multiagent_rlrm.environments.utils_envs.mdp_vi import *
from PIL import Image as PILImage
import io


# Set logging
logging.basicConfig(level=logging.INFO)

# Define available maps and experiments
AVAILABLE_MAPS = ["map0", "map1", "map2", "map3", "map4"]
AVAILABLE_EXPERIMENTS = [
    "exp0_simply",
    "exp0",
    "exp1",
    "exp2",
    "exp3",
    "exp4",
    "exp5",
    "exp6",
    "exp7",
]
AVAILABLE_ALGORITHMS = [
    "QL",
    "RMAX",
    "QRMAX",
    "QRM",
    "RMAXRM",
    "QRMAXRM",
    "QL_RS",
    "QRM_RS",
    "UCBVI",
    "OPSRL",
]  # also used to order algorithms to plot
NUM_EPISODES = 1000000000  # 1e9

# SEEDS = np.random.randint(0, 10000, size=NUM_RUNS)
# Save the seeds to a file
# np.save("seeds.npy", SEEDS)
# SEEDS = np.load("seeds.npy")

# Default map and experiment
DEFAULT_MAP = "map1"
DEFAULT_EXPERIMENT = "exp1"
DEFAULT_ALGORITHM = "QL"

CHECK_SUCCESS_THRESHOLD = config["global_settings"]["early_stop_on_success"]

USER_QUIT = False


def signal_handler(sig, frame):
    global USER_QUIT
    print("!!! User quit (CTRL-C) !!!")
    USER_QUIT = True


signal.signal(signal.SIGINT, signal_handler)


# TODO move to config.py
# Optimal values

OPTIMAL = {
    "map0;exp0_simply": 18,
    "map0;exp0": 28,
    "map1;exp1": 15,
    "map1;exp2": 29,
    "map1;exp3": 29,
    "map1;exp4": 30,
    "map1;exp5": 55,
    "map1;exp6": 75,
    "map2;exp1": 48,
    "map2;exp2": 98,
    "map2;exp3": 106,
    "map2;exp4": 81,
    "map2;exp5": 152,
    "map2;exp6": 212,
    "map3;exp1": 50,
    "map3;exp2": 54,
    "map3;exp3": 98,
    "map3;exp4": 108,
    "map3;exp5": 206,
    "map3;exp6": 272,
    "map4;exp7": 1000,
}


def parse_arguments(default_map, default_experiment, default_algorithm):
    """
    Parse command line arguments for selecting map, experiment, and algorithm.
    Allows for default values to be used if no arguments are given.

    :param default_map: Default map to use if not provided in arguments.
    :param default_experiment: Default experiment to use if not provided in arguments.
    :param default_algorithm: Default algorithm to use if not provided in arguments.
    :return: Parsed arguments containing selected map, experiment, algorithm, and rendering option.
    """
    parser = argparse.ArgumentParser(
        description="Run experiments with selected map, experiment, and algorithm."
    )
    parser.add_argument(
        "--sweep_id", type=str, default=None, help="Sweep ID to use for this run"
    )

    # Add arguments for map, experiment, and algorithm
    parser.add_argument(
        "--map",
        type=str,
        choices=AVAILABLE_MAPS,
        default=default_map,
        help="Map to use for the experiment. Choose from: "
        + ", ".join(AVAILABLE_MAPS)
        + ". Default is "
        + default_map,
    )

    parser.add_argument(
        "--experiment",
        type=str,
        choices=AVAILABLE_EXPERIMENTS,
        default=default_experiment,
        help="Experiment to run. Choose from: "
        + ", ".join(AVAILABLE_EXPERIMENTS)
        + ". Default is "
        + default_experiment,
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=AVAILABLE_ALGORITHMS,
        default=default_algorithm,
        help="Algorithm to use. Choose from: "
        + ", ".join(AVAILABLE_ALGORITHMS)
        + ". Default is "
        + default_algorithm,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Specify a custom seed for the experiment. Default 100.",
    )

    parser.add_argument(
        "--steps",
        type=float,
        default=None,
        help="Number of training steps. Default None (use value in config.py)",
    )

    parser.add_argument(
        "--save",
        type=str,
        help="Specify a file path to save the trained policy.",
    )

    parser.add_argument(
        "--load",
        type=str,
        help="Specify a file path to load a pre-trained policy.",
    )

    parser.add_argument(
        "--play",
        type=int,
        nargs="?",  # Allows optional integer input
        const=100,  # Default to 100 if no number is provided
        help="Execute the loaded policy for a given number of episodes (default is 100 if no number is provided).",
    )

    parser.add_argument(
        "--render",
        action="store_true",  # This makes --render a boolean flag, default is False
        help="Enable rendering of the environment. Disabled by default.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        help="Discount factor. Default is 0.9",
    )

    parser.add_argument(
        "--stochastic",
        default=False,
        action="store_true",
        help="Stochastic actions. Default is [False]",
    )

    parser.add_argument(
        "--all-slip",  # attiva l’all‑slip
        action="store_true",
        help="(solo con --stochastic) azioni slittano in tutte le direzioni",
    )

    parser.add_argument(
        "--highprob",
        type=float,
        default=0.8,
        help="Prob of action nominal outcome. Default is 0.8",
    )

    parser.add_argument(
        "--VIdelta",
        type=float,
        default=1e-2,
        help="Delta threshold for VI. Default is 1e-3",
    )

    parser.add_argument(
        "--VIdeltarel",
        default=True,
        action="store_true",
        help="If delta threshod for VI is relative. Default is [True]",
    )

    parser.add_argument(
        "--Kthreshold",
        type=int,
        default=39,
        help="Known threshold for RMAX/QRMAX (stochastic env). Default is 30. Set to 1 in deterministic environments.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate for Q-learning/QRM (stochastic env). Default is 0.1. Set to 1.0 in deterministic environments.",
    )

    parser.add_argument(
        "--wandb",
        default=False,
        action="store_true",
        help="Use wand logging",
    )

    parser.add_argument(
        "--generate_heatmap",
        action="store_true",
        help="Generate heatmaps after training. Default is False.",
    )

    parser.add_argument(
        "--early_stop",
        action="store_true",
        help="Enable early stopping based on performance. If set, training may stop before reaching max steps.",
    )

    parser.add_argument(
        "--eval",
        type=int,
        default=1000,
        help="Evaluate steps interval during training (default:1000) set to None for no evaluation).",
    )

    parser.add_argument(
        "--vi_cache",
        action="store_true",
        help="If set, use the VI cache (if available) instead of computing VI from scratch. Default is not to use the cache.",
    )

    args = parser.parse_args()

    if not args.stochastic:
        args.Kthreshold = 1
        args.learning_rate = 1.0

    return args


def getenvstr(args):

    global_config = config["global_settings"]

    plants_penalty_value = global_config["plants_penalty"]
    wall_penalty_value = global_config["wall_penalty"]
    terminate_on_plants = global_config["terminate_on_plants"]
    terminate_hit_walls = global_config["terminate_hit_walls"]

    envstr = f"Stoc({args.highprob:.2f})" if args.stochastic else "Det"
    envstr += "-"
    envstr += "W" if terminate_hit_walls else "w"
    envstr += f"({wall_penalty_value})" if wall_penalty_value != 0 else ""
    envstr += "P" if terminate_on_plants else "p"
    envstr += f"({plants_penalty_value})" if plants_penalty_value != 0 else ""
    args.env_str = envstr


def setup_environment(args):
    """
    Configures the environment and agents based on the selected map, experiment, algorithm, and seed.
    Initializes the environment, agents, and their reward machines.

    :param map_name: The name of the map.
    :param args.experiment: The selected experiment to configure.
    :param args.algorithm: The name of the algorithm to use.
    :param seed: The random seed for reproducibility.
    :return: The configured RMEnvironmentWrapper instance along with coordinates, office walls, and goals.
    """
    map_config = config["maps"][args.map]
    global_config = config["global_settings"]
    experiment_config = get_experiment_for_map(args.map, args.experiment)

    # Generate the environment and RewardMachine
    coordinates, goals, office_walls = parse_office_world(map_config["layout"])
    position_map = map_config["position_map"](coordinates, goals)
    transitions = experiment_config["transitions"]
    office_walls += [(b, a) for (a, b) in office_walls]

    # Initialize the environment
    env = MultiAgentOfficeWorld(
        width=map_config["grid_size"][1],
        height=map_config["grid_size"][0],
        plants=coordinates["plant"],
        coffee=coordinates["coffee"],
        letters=coordinates["letter"],
        walls=office_walls,
        plants_penalty_value=global_config["plants_penalty"],
        wall_penalty_value=global_config["wall_penalty"],
        terminate_on_plants=global_config["terminate_on_plants"],
        terminate_hit_walls=global_config["terminate_hit_walls"],
    )
    env.stochastic_env = global_config["stochastic_env"]
    env.all_slip = args.all_slip

    env.stochastic = args.stochastic
    env.high_prob = (
        args.highprob
    )  # prob of nominal outcome of actions in stochastic env

    # Initialize the agent
    agent_config = map_config["agents"][0]

    agent = AgentRL(f'{agent_config["name"]}_{args.experiment}_{args.algorithm}', env)
    agent.set_initial_position(*agent_config["position"])
    agent.add_state_encoder(StateEncoderOfficeWorld(agent))

    event_detector = PositionEventDetector(position_map)
    agent.set_reward_machine(RewardMachine(transitions, event_detector))

    # Se l'algoritmo scelto richiede il reward shaping, lo configuriamo
    if args.algorithm == "QRM_RS":
        # Puoi usare args.gamma per rs_gamma oppure aggiungere un nuovo parametro
        agent.get_reward_machine().add_reward_shaping(args.gamma, args.gamma)
        # agent.get_reward_machine().add_distance_reward_shaping(args.gamma, args.gamma)

    # Algorithm mapping
    algorithm_map = {
        "QRMAX": lambda state_size, rm_states, seed, stoc: create_qrmax(
            state_size, rm_states, seed, stochastic=stoc
        ),
        "QRM": lambda state_size, seed, stoc: create_qlearning(
            state_size, seed, use_qrm=True, stochastic=stoc, use_rsh=False
        ),
        "QL": lambda state_size, seed, stoc: create_qlearning(
            state_size, seed, use_qrm=False, stochastic=stoc, use_rsh=False
        ),
        "RMAX": lambda state_size, seed, stoc: create_rmax(
            state_size, seed, stochastic=stoc
        ),
        "RMAXRM": lambda state_size, seed, stoc: create_rmax(
            state_size, seed, stochastic=stoc, use_qrm=True
        ),
        "QRMAXRM": lambda state_size, rm_states, seed, stoc: create_qrmax(
            state_size, rm_states, seed, stochastic=stoc, use_qrm=True
        ),
        "QL_RS": lambda state_size, seed, stoc: create_qlearning(
            state_size, seed, use_qrm=False, stochastic=stoc, use_rsh=True
        ),
        "QRM_RS": lambda state_size, seed, stoc: create_qlearning(
            state_size, seed, use_qrm=True, stochastic=stoc, use_rsh=True
        ),
        "UCBVI": lambda state_size, seed, stoc: create_ucbvi(
            state_size, seed, stochastic=stoc
        ),
        "OPSRL": lambda state_size, seed, stoc: create_opsrl(
            state_size, seed, stochastic=stoc
        ),
    }

    if args.algorithm in algorithm_map:
        algo_creator = algorithm_map[args.algorithm]
    else:
        raise ValueError(f"Algorithm '{args.algorithm}' unknown.")

    assign_learning_algorithm(env, agent, algo_creator, args.seed)

    # Register symbolic movement actions via the Office World encoder
    # (the environment handles dynamics in `apply_action`).
    agent.add_action_encoder(ActionEncoderOfficeWorld(agent))

    env.add_agent(agent)
    rm_env = RMEnvironmentWrapper(env, [agent])

    return rm_env, coordinates, office_walls, goals


def initialize_renderer(environment, coordinates, office_walls, goals):
    """
    Initializes and configures the graphical renderer for the environment.
    Sets up the environment visualization, including agents, objects, and goals.

    :param environment: The environment instance to render.
    :param coordinates: Object coordinates in the environment.
    :param office_walls: Positions of office walls in the environment.
    :param goals: Positions of goals in the environment.
    :return: Configured renderer instance.
    """
    object_positions = {
        "plant": coordinates["plant"],
        "coffee": coordinates["coffee"],
        "letter": coordinates["letter"],
        "office_walls": office_walls,
    }
    renderer = EnvironmentRenderer(
        grid_width=environment.grid_width,
        grid_height=environment.grid_height,
        agents=environment.agents,
        object_positions=object_positions,
        goals=goals,
    )
    renderer.init_pygame()
    return renderer


def assign_learning_algorithm(env, agent, algorithm_creator, seed):
    """
    Assigns a learning algorithm to an agent within the environment.
    Based on the agent's Reward Machine state space and the provided algorithm creator function.

    :param env: The environment instance.
    :param agent: The agent to assign the algorithm.
    :param algorithm_creator: Function to create the learning algorithm.
    :param seed: Random seed for reproducibility.
    """
    state_size = (
        env.grid_width * env.grid_height * agent.get_reward_machine().numbers_state()
    )
    # Get the number of states in the Reward Machine
    rm_states_count = agent.get_reward_machine().numbers_state()

    # Call the algorithm_creator with appropriate arguments based on the type
    if "QRMAX" in agent.name:
        algorithm = algorithm_creator(state_size, rm_states_count, seed, env.stochastic)
    else:
        algorithm = algorithm_creator(state_size, seed, env.stochastic)

    agent.set_learning_algorithm(algorithm)
    agent.get_learning_algorithm().learn_init()


def create_rmax(state_size, seed, stochastic=False, use_qrm=False):
    """
    Creates an instance of the RMax learning algorithm with specific parameters.

    :param state_size: The size of the state space.
    :param seed: Random seed for reproducibility.
    :param stochastic: stochastic env.
    :return: Configured RMax instance.
    """
    return RMax(
        state_space_size=state_size,
        action_space_size=4,
        s_a_threshold=args.Kthreshold,
        max_reward=1.0,
        gamma=args.gamma,
        epsilon_one=0.99,
        VI_delta=args.VIdelta,
        VI_delta_rel=args.VIdeltarel,
        seed=seed,
        use_qrm=use_qrm,
    )


def create_qlearning(
    state_size,
    seed,
    stochastic=False,
    use_qrm=False,
    use_rsh=False,
):
    """
    Creates an instance of the QLearning algorithm with specific parameters.

    :param state_size: The size of the state space.
    :param seed: Random seed for reproducibility.
    :param use_qrm: Boolean indicating whether to use QRM or regular QLearning.
    :param stochastic: stochastic env.
    :return: Configured QLearning instance.
    """
    return QLearning(
        state_space_size=state_size,
        action_space_size=4,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        action_selection="greedy",
        epsilon_start=0.1,
        epsilon_end=0.1,
        epsilon_decay=0.9995,
        qtable_init=2,
        use_qrm=use_qrm,
        use_rsh=use_rsh,
        seed=seed,
    )


def create_qrmax(state_size, q_space_size, seed, stochastic=False, use_qrm=False):
    """
    Creates an instance of the QRMax learning algorithm with specific parameters.

    :param state_size: The size of the state space.
    :param q_space_size: The number of states in the Reward Machine.
    :param seed: Random seed for reproducibility.
    :param stochastic: stochastic env.
    :return: Configured QRMax instance.
    """
    return QRMax_v2(
        state_space_size=state_size,
        action_space_size=4,
        gamma=args.gamma,
        VI_delta=args.VIdelta,
        VI_delta_rel=args.VIdeltarel,
        q_space_size=q_space_size,  # Use RM states count here
        nsamplesTE=args.Kthreshold,  # Transition Environment - threshold to consider a transition (s, a) known in the environment
        nsamplesRE=args.Kthreshold,
        # Reward Environment - threshold to consider the reward associated with a pair (s, a) known in the environment
        nsamplesTQ=1,
        # Transition for Q - threshold to consider a state transition of the Reward Machine automaton (q, s') known given a pair (s, a)
        nsamplesRQ=1,
        # Reward for Q - threshold to consider the reward associated with a transition of the Reward Machine automaton (q, s', q')
        seed=seed,
        use_qrm=use_qrm,
    )


def create_ucbvi(state_size, seed, stochastic=False):
    # puoi sostituire i valori fissi con quelli di args.* se ti servono
    return UCBVI(
        state_space_size=state_size,
        action_space_size=4,
        ep_len=150,  # oppure args.ep_len
        # gamma=args.gamma,       # se vuoi rispettare il CLI
        bonus_type="bernstein",  # simplified_bernstein, bernstein, hoeffding
        bonus_scaling=1,
        reward_free=False,
        stage_dependent=False,
        real_time_dp=False,
        seed=seed,
        # reward_range=(0, 1),
        debug_interval=25,
    )


def create_opsrl(state_size, seed, stochastic=False):
    return OPSRL(
        state_space_size=state_size,
        action_space_size=4,
        ep_len=150,
        gamma=1,
        thompson_samples=1,
        bernoullized_reward=True,
        reward_free=False,
        prior_transition="uniform",  # uniform o 'optimistic'
        make_absorbing_on_done=False,  # se vuoi comportamento RMAX
        seed=seed,
    )


def initialize_wandb(args):
    """
    Initializes Weights & Biases (wandb) for experiment logging.

    :param algo: Name of the algorithm used in the experiment
    :param args.experiment: Name of the selected experiment.
    :param seed: Random seed for reproducibility.
    """
    WANDB_PROJECT = config["wandb"]["project"]
    WANDB_ENTITY = config["wandb"]["entity"]
    USE_WANDB = config["wandb"]["activate"]

    os.environ["WANDB_SILENT"] = "true"

    hyperparameters = {
        "env": args.env_str,
        "map": args.map,
        "experiment": args.experiment,
        "stochastic": args.stochastic,
        "highprob": args.highprob if args.stochastic else None,
        "algorithm": args.algorithm,
        "seed": args.seed,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate if "RMAX" not in args.algorithm else None,
        "VIdeltarel": args.VIdeltarel if "RMAX" in args.algorithm else None,
        "VIdelta": args.VIdelta if "RMAX" in args.algorithm else None,
        "Kthreshold": args.Kthreshold if "RMAX" in args.algorithm else None,
        "early_stop": args.early_stop,
        "plot_order": AVAILABLE_ALGORITHMS.index(
            args.algorithm
        ),  # used to order algorithms to plot
    }

    name = f"{args.env_str}_{args.map}_{args.experiment}"  # _{args.algorithm}_Seed_{args.seed}_Gamma_{args.gamma}_Kthreshold_{args.Kthreshold}_VIdelta_{args.VIdelta}_EarlyStop_{args.early_stop}"

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        # mode="disabled",
        name=name,
        config=hyperparameters,
        reinit=True,
    )


def log_wandb_data(
    rm_env,
    episode,
    rewards_agents,
    successi_per_agente,
    ricompense_per_episodio,
    finestra_media_mobile,
    total_step,
):
    """
    Logs data to Weights & Biases during the experiment.

    :param rm_env: The Reward Machine environment instance.
    :param episode: The current episode number.
    :param rewards_agents: Rewards obtained by each agent.
    :param successi_per_agente: Success count per agent.
    :param ricompense_per_episodio: Rewards per episode.
    :param finestra_media_mobile: Moving average window size.
    :param total_step: Total steps taken in the current run.
    """
    log_data = prepare_log_data(
        rm_env.env,
        episode,
        rewards_agents,
        successi_per_agente,
        ricompense_per_episodio,
        finestra_media_mobile,
    )
    log_data.update({f"Steps_total": total_step})

    wandb.log(log_data, step=episode)  # TODO use total_step here!!!


def test_policy(args, rm_env, episode, train_steps, play=False, optimal_steps=30):
    """
    Tests the current policy, logging results. Uses a different number of episodes depending on whether in play mode.

    :param rm_env: The Reward Machine environment instance.
    :param episode: Current episode number.
    :param play: Indicates the number of episodes to test during play mode, defaults to config settings if False.
    :param optimal_steps: Expected optimal number of steps.
    :return: The success rate and average timesteps during testing.
    """

    # Determine the number of test episodes based on env and whether play is specified
    if play:
        num_test_episodes = play
    elif not rm_env.env.stochastic:
        num_test_episodes = 1
    else:
        num_test_episodes = config["global_settings"]["num_test_optimal_policy"]

    (
        success_rate_per_agente,
        moving_averages,
        avg_timesteps_per_agente,
        std_timesteps_per_agente,
        avg_reward_per_agente,
        std_reward_per_agente,
        avg_arps_per_agente,
    ) = test_policy_optima(
        rm_env,
        episodi_test=num_test_episodes,
        optimal_steps=optimal_steps,
        gamma=args.gamma,
        test_deterministic=True,
    )
    ag_name = rm_env.agents[0].name

    avg_timesteps = avg_timesteps_per_agente[ag_name]
    avg_reward = avg_reward_per_agente[ag_name]
    success_rate = success_rate_per_agente[ag_name]

    avg_timesteps_stoc = None
    avg_reward_stoc = None
    success_rate_stoc = None

    if rm_env.env.stochastic:
        (
            success_rate_per_agente,
            moving_averages,
            avg_timesteps_per_agente,
            std_timesteps_per_agente,
            avg_reward_per_agente,
            std_reward_per_agente,
            avg_arps_per_agente,
        ) = test_policy_optima(
            rm_env,
            episodi_test=num_test_episodes,
            optimal_steps=optimal_steps,
            gamma=args.gamma,
            test_deterministic=False,
        )
        avg_timesteps_stoc = avg_timesteps_per_agente[ag_name]
        avg_reward_stoc = avg_reward_per_agente[ag_name]
        success_rate_stoc = success_rate_per_agente[ag_name]

    """print(
        "success_rate: ",
        success_rate_per_agente,
        "episode: ",
        episode,
        "timestep: ",
        average_timesteps,
        "start_pos: ",
        rm_env.agents[0].get_position(),
        "reward: ",
        avg_reward_per_agente,
    )"""
    # Check if all agents have a success rate of 0

    save_heatmap_during_training = config["heat_map"].get(
        "save_heatmap_during_training", False
    )
    heatmap_start_episode = config["heat_map"].get("heatmap_start_episode", 0)
    if (
        save_heatmap_during_training
        and episode >= heatmap_start_episode
        and all(rate == 0 for rate in success_rate_per_agente.values())
    ):
        generate_and_save_heatmaps(rm_env, args, episode=episode)

    """
    for _, rewag in success_rate_per_agente.items():
        if rewag > 0:
            logging.info(
                    f"[{episode}] {args.stochastic};{args.map};{args.experiment} Test success rate: {success_rate_per_agente} - avg timesteps {average_timesteps} - avg reward {avg_reward_per_agente} - avg arps {avg_arps_per_agente}" )
    """

    # Se non siamo in modalità play, logghiamo i dati su wandb
    if not play and args.wandb:
        log_data = {}

        if True:  # TODO CHECK new log info
            log_data["test_avg_reward"] = avg_reward
            log_data["test_success_rate"] = success_rate
            log_data["test_avg_timesteps"] = avg_timesteps
            if avg_reward_stoc is not None:
                log_data["test_avg_reward_stoc"] = avg_reward_stoc
                log_data["test_success_rate_stoc"] = success_rate_stoc
                log_data["test_avg_timesteps_stoc"] = avg_timesteps_stoc

        else:  # old log info
            for ag_name, arps in avg_arps_per_agente.items():
                log_data[f"avg_arps_{ag_name}"] = arps

            for ag_name, i_success_rate in success_rate_per_agente.items():
                log_data[f"success_rate_optima_{ag_name}"] = i_success_rate

            for ag_name, i_avg_reward in avg_reward_per_agente.items():
                log_data[f"avg_reward_optima_{ag_name}"] = i_avg_reward

            for ag_name, i_avg_timesteps in avg_timesteps_per_agente.items():
                log_data[f"avg_timesteps_optima_{ag_name}"] = i_avg_timesteps

        wandb.log(log_data, step=train_steps)

    return (
        avg_reward,
        success_rate,
        avg_timesteps,
        avg_reward_stoc,
        success_rate_stoc,
        avg_timesteps_stoc,
    )


def initialize_experiment_metrics(agents):
    """
    Initializes metrics to track the experiment's progress, including successes, rewards, and actions.

    :param agents: List of agents in the environment.
    :return: Initialized dictionaries for tracking metrics.
    """
    successi_per_agente = {agent.name: 0 for agent in agents}
    ricompense_per_episodio = {agent.name: [] for agent in agents}
    actions_log = {agent.name: [] for agent in agents}
    finestra_media_mobile = 1000
    return (
        successi_per_agente,
        ricompense_per_episodio,
        actions_log,
        finestra_media_mobile,
    )


def generate_and_save_heatmaps(rm_env, args, episode=None):
    """
    Generates and saves heatmaps for the agents in the environment.

    :param rm_env: The RMEnvironmentWrapper instance containing the agents.
    :param args: The parsed command-line arguments.
    :param episode: The current episode number (optional, used for filename).
    """
    if not (args.generate_heatmap or args.render):
        return

    # Access the Q-tables directly from the agents
    q_tables_data = extract_q_tables(rm_env.agents)

    # Obtain the grid dimensions
    grid_width = rm_env.env.grid_width
    grid_height = rm_env.env.grid_height
    grid_dims = (grid_height, grid_width)

    # Get map configuration and parse the office world
    map_config = config["maps"][args.map]
    coordinates, goals, office_walls = parse_office_world(map_config["layout"])

    # Generate heatmaps and get figures
    heatmap_figures = generate_heatmaps_for_agents(
        rm_env.agents,
        q_tables_data,
        grid_dims=grid_dims,
        walls=office_walls,
        plants=coordinates.get("plant", []),
    )

    # Define the output directory
    output_dir = "heatmaps"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    for agent_name, fig in heatmap_figures:
        if args.wandb:
            wandb.log({f"Heatmap_{agent_name}": wandb.Image(fig)})
        else:
            if episode is not None:
                file_path = os.path.join(
                    output_dir, f"Heatmap_{agent_name}_{episode}.png"
                )
            else:
                file_path = os.path.join(output_dir, f"Heatmap_{agent_name}.png")
            fig.savefig(file_path)  # Save the figure as a PNG image
            logging.info(f"Saved heatmap for {agent_name} to {file_path}")
        plt.close(fig)  # Close the figure to free memory


def compute_and_plot_vi_policy_multi(rm_env, office_walls, coordinates, goals, args):
    """
    Computes the optimal policy for each agent using Value Iteration (VI).
    Optionally loads/saves results from a cache file to speed up repeated tests.

    Generates and saves heatmaps for the value function and the policy.

    Returns:
      - A dictionary {agent.name -> (value_function, policy_vi, q_table)} containing
        the computed VI results for each agent.
    """
    cache_dir = "vi_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # 1. Name of the file to save to or load VI results from.
    # vi_cache_file = os.path.join(cache_dir, f"vi_{args.map}_{args.experiment}.pkl") #altrimenti rimuovere il nome dell'algo dal nome dell'agente

    # Includi l'algoritmo nel nome del file di cache
    vi_cache_file = os.path.join(
        cache_dir, f"vi_{args.map}_{args.experiment}_{args.algorithm}.pkl"
    )

    # 2. If the cache file exists, we try to load the results
    if args.vi_cache and os.path.exists(vi_cache_file):
        with open(vi_cache_file, "rb") as f:
            results_vi = pickle.load(f)
        logging.info(f"[VI] Loaded results from cache file: {vi_cache_file}")

    else:
        # 2a. If it does not exist, we build the MDP models and calculate the VI
        logging.info("[VI] No cache file found. Calculating and saving results...")

        rm_env.reset(args.seed)
        # Build the MDP with steps, as desired
        P_all, num_states_all, num_actions_all = rm_env.get_mdp(args.seed)

        results_vi = {}
        # For each agent we calculate the VI
        for agent in rm_env.agents:
            P = P_all[agent.name]
            nS = num_states_all[agent.name]
            nA = num_actions_all[agent.name]

            V, policy_vi, Q = value_iteration(
                P,
                nS,
                nA,
                gamma=args.gamma,
                theta=1e-4,
                delta_rel=args.VIdeltarel,
            )

            # Saving results in 'results_vi'
            results_vi[agent.name] = (V, policy_vi, Q)

        # 2b. We save the VI results to the cache file
        with open(vi_cache_file, "wb") as f:
            pickle.dump(results_vi, f)
        logging.info(f"[VI] Saved results to file cache: {vi_cache_file}")

    # If neither generate_heatmap nor render are active, do not generate images
    if not (args.generate_heatmap or args.render):
        return results_vi

    # 3. Heatmap display and figure saving
    for agent in rm_env.agents:
        (V, policy_vi, Q) = results_vi[agent.name]

        rm = agent.get_reward_machine()
        grid_width = rm_env.env.grid_width
        grid_height = rm_env.env.grid_height

        fig_vi = generate_value_policy_heatmap(
            V,
            policy_vi,
            grid_dims=(grid_height, grid_width),
            rm=rm,
            walls=office_walls,
            plants=coordinates["plant"],
            goals=goals,
            coordinates=coordinates,
        )
        fig_vf = generate_value_heatmap(
            V,
            (grid_height, grid_width),
            rm,
            walls=office_walls,
            plants=coordinates["plant"],
            goals=goals,
        )

        # Directory di output
        output_dir = "heatmaps"
        os.makedirs(output_dir, exist_ok=True)

        # # If you use wandb, load the figures to wandb; otherwise to PNG files
        if args.wandb:
            wandb.log({f"{agent.name}_value_policy_vi_heatmap": wandb.Image(fig_vi)})
            wandb.log({f"{agent.name}_value_function_vi_heatmap": wandb.Image(fig_vf)})
        else:
            file_path_vi = os.path.join(
                output_dir, f"{agent.name}_value_policy_vi_heatmap.png"
            )
            file_path_vf = os.path.join(
                output_dir, f"{agent.name}_value_function_vi_heatmap.png"
            )

            fig_vi.savefig(file_path_vi, dpi=300)
            logging.info(f"[VI] Saved {agent.name} policy heatmap to {file_path_vi}")

            fig_vf.savefig(file_path_vf, dpi=300)
            logging.info(f"[VI] Saved {agent.name} value heatmap to {file_path_vf}")

        plt.close(fig_vi)
        plt.close(fig_vf)

    return results_vi


def test_vi_policy_multi(
    rm_env,
    policy_vi,
    gamma=0.95,
    optimal_steps=98,
    episodes_test=100,
    test_deterministic=False,
):
    """
    Tests the Value Iteration (VI) policy over multiple episodes for all agents.

    Parameters:
      - rm_env: The environment to test.
      - policy_vi: The VI policy dictionary {agent.name -> policy}.
      - gamma: Discount factor for rewards.
      - optimal_steps: Expected optimal steps for ARPS calculation.
      - episodes_test: Number of testing episodes.
      - test_deterministic: If True, forces the environment to be deterministic.

    Returns:
      - The test results as computed by `test_policy_opt_multi`.
    """
    results_vi = test_policy_opt_multi(
        rm_env,
        policy_vi,
        episodes_test=episodes_test,
        gamma=gamma,
        optimal_steps=optimal_steps,
        test_deterministic=test_deterministic,
    )
    return results_vi


def test_rl_policy_multi(
    rm_env, gamma=0.95, optimal_steps=98, episodes_test=10000, test_deterministic=False
):
    """
    Extracts the policy from the Reinforcement Learning (RL) Q-table,
    tests it over multiple episodes, and returns the test results.

    Parameters:
      - rm_env: The environment containing agents with learned Q-tables.
      - gamma: Discount factor for rewards.
      - optimal_steps: Expected optimal steps for ARPS calculation.
      - episodes_test: Number of testing episodes.
      - test_deterministic: If True, forces the environment to be deterministic.

    Returns:
      - The test results as computed by `test_policy_opt_multi`.
    """
    # q_table_agent = extract_q_tables(rm_env.agents)
    # q_table = q_table_agent[f"q_table_{rm_env.agents[0].name}"]
    policy_rl = extract_policy_from_qtable(agent)

    results_rl = test_policy_opt_multi(
        rm_env,
        policy_rl,
        episodes_test=episodes_test,
        gamma=gamma,
        optimal_steps=optimal_steps,
        test_deterministic=test_deterministic,
    )

    return results_rl


import io
import numpy as np
import matplotlib.pyplot as plt
import logging
import wandb
from PIL import Image as PILImage


def perform_and_log_ttest(ricompense_vi, ricompense_rl, agente_name, args):
    """
    Esegue un t-test fra due serie di ricompense (VI vs RL)
    e logga i risultati (plot + statistiche) su wandb.
    """

    if not args.stochastic:
        logging.info("Deterministic environment, skip T-test to avoid var=0.")
        return 0.0, 1.0, 0.0, 0.0, 0.0, 0.0

    statistic, pvalue = perform_ttest(
        ricompense_vi, ricompense_rl, agente_name=agente_name
    )

    vi_rewards_list = np.concatenate(list(ricompense_vi.values()))
    rl_rewards_list = np.concatenate(list(ricompense_rl.values()))

    mean_vi = np.mean(vi_rewards_list)
    std_vi = np.std(vi_rewards_list)
    mean_rl = np.mean(rl_rewards_list)
    std_rl = np.std(rl_rewards_list)

    gamma = args.gamma
    aname = f"{args.algorithm}({gamma})"

    # Log su console
    logging.info(f"T-test statistic: {statistic}, p-value: {pvalue}")
    logging.info(f"Mean VI({gamma}): {mean_vi}, Std VI({gamma}): {std_vi}")
    logging.info(f"Mean {aname}: {mean_rl}, Std {aname}: {std_rl}")

    # Se non usiamo wandb, basta così
    if not args.wandb:
        return statistic, pvalue, mean_vi, std_vi, mean_rl, std_rl

    # Altrimenti costruiamo i plot e li logghiamo su wandb
    # 1) Bar chart delle Mean
    fig_means = plt.figure(figsize=(6, 4))
    plt.bar(
        [f"Mean VI({gamma})", f"Mean {aname}"],
        [mean_vi, mean_rl],
        color=["blue", "orange"],
    )
    plt.ylabel("Mean Reward")
    plt.title(f"Mean Rewards Comparison (Agent: {agente_name})")
    plt.tight_layout()

    # Salvataggio in un buffer + conversione in PIL
    means_buf = io.BytesIO()
    fig_means.savefig(means_buf, format="png")
    plt.close(fig_means)
    means_buf.seek(0)
    means_img = PILImage.open(means_buf)

    # 2) Bar chart delle Std
    fig_stds = plt.figure(figsize=(6, 4))
    plt.bar(
        [f"Std VI({gamma})", f"Std {aname}"], [std_vi, std_rl], color=["blue", "orange"]
    )
    plt.ylabel("Standard Deviation")
    plt.title(f"Standard Deviation Comparison (Agent: {agente_name})")
    plt.tight_layout()

    stds_buf = io.BytesIO()
    fig_stds.savefig(stds_buf, format="png")
    plt.close(fig_stds)
    stds_buf.seek(0)
    stds_img = PILImage.open(stds_buf)

    # 3) Bar chart per T-test e P-value
    fig_ttest = plt.figure(figsize=(6, 4))
    plt.bar(
        ["T-test Statistic", "P-value"], [statistic, pvalue], color=["green", "red"]
    )
    plt.ylabel("Value")
    plt.title(f"T-test Results (Agent: {agente_name})")
    plt.tight_layout()

    ttest_buf = io.BytesIO()
    fig_ttest.savefig(ttest_buf, format="png")
    plt.close(fig_ttest)
    ttest_buf.seek(0)
    ttest_img = PILImage.open(ttest_buf)

    # Log finale su wandb passando le immagini PIL
    wandb.log(
        {
            "Mean Comparison": wandb.Image(means_img),
            "Standard Deviation Comparison": wandb.Image(stds_img),
            "T-test Results": wandb.Image(ttest_img),
        }
    )

    return statistic, pvalue, mean_vi, std_vi, mean_rl, std_rl


def do_ttest_vi_vs_rl(rm_env, vi_results, args, episodes_test=100):
    """
    Compares the current RL policy with the VI policy (provided in vi_results) for *all* agents
    in the `rm_env` wrapper. Runs multi-episode tests and computes a T-test for each agent.

    Parameters:
    ----------
    rm_env : RMEnvironmentWrapper
        The environment (wrapper) containing references to RL agents and the base environment.
    vi_results : dict
        Dictionary {agent.name -> (V, policy_vi, Q_vi)} obtained from a function
        such as `compute_and_plot_vi_policy_multi(...)`.
    args : argparse.Namespace
        Global configuration/arguments (gamma, stochastic, etc.).
    episodes_test : int
        Number of test episodes to run for evaluating RL and VI.

    Returns:
    --------
    results_ttest : dict
        Dictionary { agent.name -> (statistic, pvalue, mean_vi, std_vi, mean_rl, std_rl) }
        for each agent, as the result of the VI vs RL T-test.

    Notes:
    -----
    - If the environment is deterministic, the T-test is skipped.
    - If `vi_results` is not available, it must be computed or loaded from a file beforehand.
    """

    # 1) Costruisce un dizionario di policy VI: { agent.name -> policy_vi }
    policy_vi_dict = {}
    for agent_name, (V, policy_vi, Q) in vi_results.items():
        policy_vi_dict[agent_name] = policy_vi

    # 2) Builds a dictionary of RL policies (extracted from Q-table or other)
    policy_rl_dict = {}
    for agent in rm_env.agents:
        policy_rl = extract_policy_from_qtable(agent)
        policy_rl_dict[agent.name] = policy_rl

    # 3) Multi-agent test for VI policy
    # test_policy_opt_multi(...) returns a tuple,
    # of which the eighth element (index 7) is the dictionary
    # {agent.name -> reward_list}.
    results_vi = test_policy_opt_multi(
        rm_env=rm_env,
        policy_dict=policy_vi_dict,
        episodes_test=episodes_test,
        gamma=args.gamma,
        optimal_steps=OPTIMAL.get(f"{args.map};{args.experiment}", 30),
        test_deterministic=not args.stochastic,
    )
    ricompense_vi_all = results_vi[7]  # dict {ag.name -> [rewards_per_episode]}

    # 4) Multi-agent testing for RL policy
    results_rl = test_policy_opt_multi(
        rm_env=rm_env,
        policy_dict=policy_rl_dict,
        episodes_test=episodes_test,
        gamma=args.gamma,
        optimal_steps=OPTIMAL.get(f"{args.map};{args.experiment}", 30),
        test_deterministic=not args.stochastic,
    )
    ricompense_rl_all = results_rl[7]

    # 5) Run VI vs. RL T-test for each agent and collect the results
    results_ttest = {}
    for agent in rm_env.agents:
        ag_name = agent.name

        # If the environment is deterministic, skip by default
        if not args.stochastic:
            logging.info(f"Skipping T-test in deterministic env for agent {ag_name}.")
            results_ttest[ag_name] = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
            continue

        # Build the {ag_name -> rewards_list} dictionaries for T-test
        vi_rewards_dict = {ag_name: ricompense_vi_all[ag_name]}
        rl_rewards_dict = {ag_name: ricompense_rl_all[ag_name]}

        # Call the function that does the T-test and logs the results
        statistic, pvalue, mean_vi, std_vi, mean_rl, std_rl = perform_and_log_ttest(
            vi_rewards_dict, rl_rewards_dict, ag_name, args
        )
        results_ttest[ag_name] = (statistic, pvalue, mean_vi, std_vi, mean_rl, std_rl)

    return results_ttest


def run_post_processing_multi(
    rm_env, coordinates, office_walls, goals, args, opt, episodes_test=10000
):
    """
    Executes post-processing for all agents in the environment, including:

    1. Computing and plotting VI policies for all agents, saving heatmaps.
    2. Testing VI policies and extracting performance metrics.
    3. Extracting RL policies from agent Q-tables.
    4. Testing RL policies and comparing with VI policies using t-tests.
    5. Logging and saving all results for analysis.

    Parameters:
      - rm_env: The environment containing agents.
      - coordinates: Coordinates of key environment features (e.g., goals, obstacles).
      - office_walls: Positions of walls in the environment.
      - goals: Goal states for the agents.
      - args: Experiment settings.
      - opt: Optimal steps used as a baseline for performance metrics.
      - episodes_test: Number of episodes for policy evaluation.

    Returns:
      - A dictionary containing t-test results for each agent.
    """
    # Determina se testare in modalità deterministica o meno
    test_deterministic = not args.stochastic

    # 1) Calculate the multi-agent VI policy and produce any heatmaps
    vi_results = compute_and_plot_vi_policy_multi(
        rm_env, office_walls, coordinates, goals, args
    )
    # vi_results è un dict: { agent.name -> (V, policy_vi, Q) }

    # 2) Build a VI policy dictionary: { agent.name -> policy_vi }
    policy_vi_dict = {}
    for agent_name, (V, policy_vi, Q) in vi_results.items():
        policy_vi_dict[agent_name] = policy_vi

    # 3) Parallel VI testing for *all* agents with test_policy_opt_multi
    results_vi = test_policy_opt_multi(
        rm_env=rm_env,
        policy_dict=policy_vi_dict,  # Dizionario di policy VI
        episodes_test=episodes_test,
        gamma=args.gamma,
        optimal_steps=opt,
        test_deterministic=test_deterministic,
    )
    ricompense_vi_all = results_vi[7]  # { agent.name -> lista di ricompense }

    # 4) Builds a RL policy dictionary, extracting the RL policy from each agent's Q-table
    policy_rl_dict = {}
    for agent in rm_env.agents:
        policy_rl = extract_policy_from_qtable(agent)
        policy_rl_dict[agent.name] = policy_rl

    # 5) Multi-agent RL testing
    results_rl = test_policy_opt_multi(
        rm_env=rm_env,
        policy_dict=policy_rl_dict,  # Dizionario di policy RL
        episodes_test=episodes_test,
        gamma=args.gamma,
        optimal_steps=opt,
        test_deterministic=test_deterministic,
    )
    ricompense_rl_all = results_rl[7]

    # 6) Run the T-test for each agent and log the results
    results_ttest = {}
    for agent in rm_env.agents:
        ag_name = agent.name

        # Extract the reward list for this agent's VI and RL
        # as a dict {ag_name: rewards_list} for compatibility with perform_and_log_ttest
        vi_rewards_dict = {ag_name: ricompense_vi_all[ag_name]}
        rl_rewards_dict = {ag_name: ricompense_rl_all[ag_name]}

        # Run T-test
        statistic, pvalue, mean_vi, std_vi, mean_rl, std_rl = perform_and_log_ttest(
            vi_rewards_dict, rl_rewards_dict, ag_name, args
        )
        results_ttest[ag_name] = (statistic, pvalue, mean_vi, std_vi, mean_rl, std_rl)

    return results_ttest


def run_experiment(args):
    """
    Runs the main experiment loop for multi-agent learning, including:

    1. Setting up the environment and agents.
    2. Running training episodes with RL algorithms.
    3. Periodically testing policies and evaluating performance.
    4. Logging metrics, saving models, and generating heatmaps.
    5. Post-processing results, including VI/RL comparisons and statistical tests.

    Parameters:
      - args: Configuration arguments specifying environment settings, logging, and evaluation options.
    """
    if args.load and args.play:
        with open(args.load, "rb") as f:
            trained_policy = pickle.load(f)

    opt = OPTIMAL[f"{args.map};{args.experiment}"]

    rm_env, coordinates, office_walls, goals = setup_environment(args)

    agent_config = config["maps"][args.map]["agents"][0]
    algo_name = agent_config["algorithm"]
    if args.steps is None:
        max_step = config["global_settings"]["max_step"]
    else:
        max_step = args.steps

    vi_results = compute_and_plot_vi_policy_multi(
        rm_env, office_walls, coordinates, goals, args
    )
    # E' un dict: { agent.name -> (V, policy_vi, Q) }

    # 3) Ciclo di training RL
    dorun = True
    episode = 0
    total_step = 0

    # Initialize renderer if rendering is enabled
    if args.render:
        renderer = initialize_renderer(
            rm_env.env,
            coordinates=coordinates,
            office_walls=office_walls,
            goals=goals,
        )

    (
        successi_per_agente,
        ricompense_per_episodio,
        actions_log,
        finestra_media_mobile,
    ) = initialize_experiment_metrics(rm_env.agents)

    average_timesteps = -1
    best_timesteps = 1e9  # best value so far
    last_printed_time = 0
    last_eval_step = 0
    eval_once = False

    alg_model = rm_env.agents[0].get_learning_algorithm()
    alg_params = alg_model.param_str

    thisfiledir = os.path.dirname(
        inspect.getsourcefile(lambda: 0)
    )  # directory del file
    expname = f"{args.env_str};{args.map};{args.experiment};{args.seed:03d}"
    aname = f"{args.algorithm}({alg_params})"

    # Definisci i percorsi dei file di log
    log_train_filename = os.path.join(
        thisfiledir, "results", f"train_{expname}_{aname}.csv"
    )
    log_test_filename = os.path.join(
        thisfiledir, "results", f"test_{expname}_{aname}.csv"
    )

    # Crea la directory 'results' se non esiste
    os.makedirs(os.path.dirname(log_train_filename), exist_ok=True)

    # Apri i file di log
    log_train = open(log_train_filename, "w" if episode == 0 else "a")
    if episode == 0:
        log_train.write(f"step;{args.algorithm}({alg_params})\n")
        with open(log_test_filename, "w") as log_test:
            log_test.write(
                f"env;map;exp;alg;seed;"
                + "episode;total_steps;log(total_step);"
                + "avg_reward;success_rate;avg_length;"
                + "avg_reward_stoc;success_rate_stoc;avg_length_stoc\n"
            )
            log_test.write(
                f"{args.env_str};{args.map};{args.experiment};{args.algorithm}({alg_params});{args.seed:03d};{episode};{total_step};0;0;0;0;0;0;0\n"
            )

    start_time = time.time()
    cet = (time.time() - start_time) / 60.0

    while episode < NUM_EPISODES and total_step < max_step:

        states, infos = rm_env.reset(args.seed * 1000 + episode)
        states = copy.deepcopy(states)
        done = {a.name: False for a in rm_env.agents}
        rewards_agents = {a.name: 0 for a in rm_env.agents}

        # Render initial state if rendering is enabled
        record_episode = episode % 1000 == 0 and episode != 0
        if args.render and record_episode:
            renderer.render(episode, states)

        got_pos_reward = False  # if received a positive reward
        cum_gamma = 1.0

        while not all(done.values()):
            actions = {}
            rewards = {a.name: 0 for a in rm_env.agents}
            # infos = {a.name: {} for a in rm_env.agents}
            for ag in rm_env.agents:
                current_state = rm_env.env.get_state(ag)
                action = ag.select_action(current_state)
                actions[ag.name] = action
                # Log delle azioni nell'ultimo episodio
                update_actions_log(actions_log, actions, NUM_EPISODES)

            new_states, rewards, terminated, truncated, infos = rm_env.step(actions)

            for agent in rm_env.agents:
                # TODO CHECK fixed error!!!
                # update should consider only terminated episodes
                # truncations are not failures and should not affect policy updates
                # terminated = done[agent.name] or truncations[agent.name]

                algterm = agent.update_policy(
                    state=states[agent.name],
                    action=actions[agent.name],
                    reward=rewards[agent.name],
                    next_state=new_states[agent.name],
                    terminated=terminated[agent.name],
                    infos=infos[agent.name],
                )
                if args.early_stop and algterm:
                    dorun = False

                if rewards[agent.name] > 0:
                    # logging.info(f"  -- ag {agent.name} ep {episode} step {total_step + rm_env.env.timestep} reward {rewards[agent.name]} --")
                    got_pos_reward = True

                rewards_agents[agent.name] += cum_gamma * rewards[agent.name]

            states = copy.deepcopy(new_states)
            cum_gamma *= args.gamma

            # Render step if rendering is enabled
            if args.render and record_episode:
                renderer.render(episode, states)

            if all(terminated.values()) or all(truncated.values()) or not dorun:
                break

        # episode is terminated

        episode += 1

        total_step += rm_env.env.timestep
        cet = (time.time() - start_time) / 60.0

        # print info every minute
        if int(cet) != last_printed_time:
            last_printed_time = int(cet)
            logging.info(
                f"{args.env_str};{args.map};{args.experiment};{args.seed:03d} {args.algorithm}({alg_params}) - Steps: {total_step} {math.log10(total_step):.3f} Time: {cet:.2f} min"
            )

        if args.render and record_episode:
            renderer.save_episode(episode, wandb=wandb if args.wandb else None)
            renderer.save_episode(episode)

        update_successes(rm_env.env, rewards_agents, successi_per_agente, done)

        # training reward
        ag_name = rm_env.agents[0].name
        training_reward = rewards_agents[ag_name]
        log_train.write(f"{total_step};{training_reward}\n")

        if args.wandb:
            log_data = {"training_reward": training_reward}
            wandb.log(log_data, step=total_step)

            """ TODO CHECK OLD log data
            log_wandb_data(
                rm_env,
                episode,
                rewards_agents,
                successi_per_agente,
                ricompense_per_episodio,
                finestra_media_mobile,
                total_step,
            )
            """

        if got_pos_reward:
            (
                avg_reward,
                success_rate,
                average_timesteps,
                avg_reward_stoc,
                success_rate_stoc,
                average_timesteps_stoc,
            ) = test_policy(args, rm_env, episode, total_step, optimal_steps=opt)

            if success_rate > 0 and average_timesteps < best_timesteps:
                best_timesteps = average_timesteps
                eval_once = True

                cet = (time.time() - start_time) / 60.0

                res_str = f"{avg_reward:.6f};{success_rate:.2f};{average_timesteps:.2f}"
                if success_rate_stoc is not None:
                    res_str += f";{avg_reward_stoc:.6f};{success_rate_stoc:.2f};{average_timesteps_stoc:.2f}"
                else:
                    res_str += ";;;"

                info_str = f"{args.env_str};{args.map};{args.experiment};{args.algorithm}({alg_params});{args.seed:03d};{episode};{total_step};{math.log10(total_step):.3f};{cet:.2f};{res_str}"

                logging.info(info_str)

                res_filename = f"results_{args.map}_{args.experiment}.txt"
                with open(res_filename, "a") as file:
                    file.write(f"{info_str}\n")

                # if args.early_stop and best_timesteps == opt: #TODO Non dovrebbe servire con l'uso di pvalue
                #    dorun = False

        # periodic evaluation
        if eval_once or (
            args.eval is not None and total_step - last_eval_step > args.eval
        ):
            eval_once = False

            (
                avg_reward,
                success_rate,
                average_timesteps,
                avg_reward_stoc,
                success_rate_stoc,
                average_timesteps_stoc,
            ) = test_policy(args, rm_env, episode, total_step, optimal_steps=opt)

            res_str = f"{avg_reward:.6f};{success_rate:.2f};{average_timesteps:.2f}"
            if success_rate_stoc is not None:
                res_str += f";{avg_reward_stoc:.6f};{success_rate_stoc:.2f};{average_timesteps_stoc:.2f}"
            else:
                res_str += ";;;"

            info_str = f"{args.env_str};{args.map};{args.experiment};{args.algorithm}({alg_params});{args.seed:03d};{episode};{total_step};{math.log10(total_step):.3f};{cet:.2f};{res_str}"

            # logging.info(info_str)
            """
            res_filename = f"results_{args.map}_{args.experiment}.txt"
            with open(res_filename, "a") as file:
                file.write(f"{info_str}\n")
            """
            with open(log_test_filename, "a") as log_test:
                log_test.write(f"{info_str}\n")

            last_eval_step = total_step

            results_ttest = do_ttest_vi_vs_rl(
                rm_env=rm_env,
                vi_results=vi_results,  # Caricato o calcolato una volta all'inizio
                args=args,
                episodes_test=100,  # Numero di episodi di test
            )

            # results_ttest -> {agent.name: (stat, pval, mean_vi, std_vi, mean_rl, std_rl)}
            stop_for_pvalue = False
            for ag_name, (statistic, pvalue, mv, sv, mr, sr) in results_ttest.items():
                if (
                    args.stochastic and pvalue > 0.1
                ):  # Criterio: RL non è significativamente peggiore di VI
                    logging.info(
                        f"Early-stop triggered for agent {ag_name} (p-value={pvalue:.4f} > 0.1)."
                    )
                    stop_for_pvalue = True
                    break

            if stop_for_pvalue:
                # Interrompi il training
                dorun = False

        if total_step > max_step or USER_QUIT or (args.early_stop and not dorun):
            break

    # end run

    log_train.close()

    cet = (time.time() - start_time) / 60.0

    # Eval last policy
    print(
        f"Eval {args.env_str};{args.map};{args.experiment};{args.algorithm}({alg_params});{args.seed:03d} ..."
    )
    (
        avg_reward,
        success_rate,
        average_timesteps,
        avg_reward_stoc,
        success_rate_stoc,
        average_timesteps_stoc,
    ) = test_policy(args, rm_env, episode, total_step, optimal_steps=opt)

    res_str = f"{avg_reward:.6f};{success_rate:.2f};{average_timesteps:.2f}"
    if success_rate_stoc is not None:
        res_str += f";{avg_reward_stoc:.6f};{success_rate_stoc:.2f};{average_timesteps_stoc:.2f}"
    else:
        res_str += ";;;"

    info_str = f"{args.env_str};{args.map};{args.experiment};{args.algorithm}({alg_params});{args.seed:03d};{episode};{total_step};{math.log10(total_step):.3f};{cet:.2f};{res_str}"

    res_filename = f"results_{args.map}_{args.experiment}.txt"
    with open(res_filename, "a") as file:
        file.write(f"{info_str}\n")
    logging.info(info_str)

    # Save last model
    if args.save:
        with open(args.save, "wb") as f:
            pickle.dump(rm_env.agents[0].get_learning_algorithm(), f)

    if args.generate_heatmap:
        generate_and_save_heatmaps(rm_env, args)

    run_post_processing_multi(
        rm_env, coordinates, office_walls, goals, args, opt, episodes_test=10000
    )

    if args.wandb:
        log_data = {"training_steps": total_step, "episodes": episode}
        wandb.log(log_data)
        wandb.finish()


if __name__ == "__main__":
    # Associate the SIGINT signal handler with the signal_handler function
    signal.signal(signal.SIGINT, signal_handler)

    # Read default algorithm from configuration
    default_algorithm = config["maps"][DEFAULT_MAP]["agents"][0]["algorithm"]

    # Parse command line arguments with default values
    args = parse_arguments(DEFAULT_MAP, DEFAULT_EXPERIMENT, DEFAULT_ALGORITHM)

    getenvstr(args)

    if args.wandb:
        # Prepara gli iperparametri
        hyperparameters = {
            "environment": args.env_str,
            "map": args.map,
            "experiment": args.experiment,
            "stochastic": args.stochastic,
            "highprob": args.highprob if args.stochastic else None,
            "algorithm": args.algorithm,
            "seed": args.seed,
            "gamma": args.gamma,
            "learning_rate": args.learning_rate
            if "RMAX" not in args.algorithm
            else None,
            "VIdeltarel": args.VIdeltarel if "RMAX" in args.algorithm else None,
            "VIdelta": args.VIdelta if "RMAX" in args.algorithm else None,
            "Kthreshold": args.Kthreshold if "RMAX" in args.algorithm else None,
            "early_stop": args.early_stop,
            "render": args.render,
            "generate_heatmap": args.generate_heatmap,
            # Aggiungi altri parametri se necessario
        }

        # Inizializza wandb con la config, senza impostare ancora il nome
        wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            config=hyperparameters,
            reinit=True,
        )

        # Aggiorna args con i parametri da wandb.config
        args.map = wandb.config.get("map", args.map)
        args.experiment = wandb.config.get("experiment", args.experiment)
        args.algorithm = wandb.config.get("algorithm", args.algorithm)
        args.seed = wandb.config.get("seed", args.seed)
        args.Kthreshold = wandb.config.get("Kthreshold", args.Kthreshold)
        args.gamma = wandb.config.get("gamma", args.gamma)
        # args.VIdeltarel = wandb.config.get("VIdeltarel", args.VIdeltarel)
        args.VIdelta = wandb.config.get("VIdelta", args.VIdelta)
        args.stochastic = wandb.config.get("stochastic", args.stochastic)
        args.highprob = wandb.config.get("highprob", args.highprob)
        args.early_stop = wandb.config.get("early_stop", args.early_stop)
        args.render = wandb.config.get("render", args.render)
        args.generate_heatmap = wandb.config.get(
            "generate_heatmap", args.generate_heatmap
        )
        args.learning_rate = wandb.config.get("learning_rate", args.learning_rate)

        getenvstr(args)

        # Ora imposta il nome della run utilizzando gli args aggiornati
        wandb.run.name = f"{args.env_str}_{args.map}_{args.experiment}"
        wandb.run.save()

    os.makedirs("results", exist_ok=True)  # Create results dir

    run_experiment(args)
