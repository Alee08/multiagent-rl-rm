# config.py
from multiagent_rlrm.multi_agent.action_rl import ActionRL
from multiagent_rlrm.utils.utils import parse_office_world

# This section defines the actions available to agents, including preconditions and effects.
# Actions dictate how agents can interact with the environment.

# Actions are defined symbolically; movement dynamics are handled directly by
# the environment (see `apply_action` in `ma_office.py`).


def can_move_up(agent):
    x, y = agent.get_position()
    return (
        y < agent.ma_problem.grid_height - 1
        and ((x, y), (x, y + 1)) not in agent.ma_problem.walls
    )


def can_move_down(agent):
    x, y = agent.get_position()
    return y > 0 and ((x, y), (x, y - 1)) not in agent.ma_problem.walls


def can_move_left(agent):
    x, y = agent.get_position()
    return x > 0 and ((x, y), (x - 1, y)) not in agent.ma_problem.walls


def can_move_right(agent):
    x, y = agent.get_position()
    return (
        x < agent.ma_problem.grid_width - 1
        and ((x, y), (x + 1, y)) not in agent.ma_problem.walls
    )


# Creating actions (symbolic only; the environment handles preconditions)
move_up = ActionRL("up")
move_down = ActionRL("down")
move_left = ActionRL("left")
move_right = ActionRL("right")


config = {
    "maps": {
        "map0": {
            "layout": """
🥤 🟩 🟩 🟩 🟩 🟩 🟩 🟩 A  🥤
🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
C  🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 
🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
D  🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 B
🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
O  🟩 🟩 🟩 E  🟩 🟩 🟩 🟩 ✉️
""",
            "grid_size": (10, 10),
            "position_map": lambda coordinates, goals: {
                coordinates["coffee"][0],
                coordinates["coffee"][1],
                coordinates["letter"][0],
                goals["A"],
                goals["B"],
                goals["C"],
                goals["D"],
                goals["E"],
                goals["O"],
            },
            "max_time": None,
            "agents": [
                {
                    "algorithm": "QRMAX",  # QRMAX, QRM, QL, RMAX
                    "name": "a3",  # a1, a2, a3, a4, a5
                    "position": (0, 0),
                    "events": None,
                    "actions": [move_up, move_down, move_left, move_right],
                },
            ],
            "groups_priority": None,  # [["a1"], ["a2"], ["a3"], ["a4"]],
        },
        "map1": {
            "layout": """
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
 🟩 B  🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 C  🟩
 🟩 🟩 🟩 ⛔ 🥤 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
 ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ 
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
 🟩 🪴 🟩 ⛔ 🟩 O  🟩 ⛔ 🟩 ✉️ 🟩 ⛔ 🟩 🪴 🟩
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
 ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ 🚪 ⛔
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🥤 ⛔ 🟩 🟩 🟩
 🟩 A  🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 D  🟩
 E  🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
 """,
            "grid_size": (9, 12),
            "position_map": lambda coordinates, goals: {
                coordinates["coffee"][0],
                coordinates["coffee"][1],
                coordinates["letter"][0],
                goals["A"],
                goals["B"],
                goals["C"],
                goals["D"],
                goals["E"],
                goals["O"],
            },
            "max_time": None,
            "agents": [
                {
                    "algorithm": "QRMAX",  # QRMAX, QRM, QL, RMAX
                    "name": "a3",  # a1, a2, a3, a4, a5
                    "position": (2, 7),
                    "events": None,
                    "actions": [move_up, move_down, move_left, move_right],
                },
            ],
            "groups_priority": None,  # [["a1"], ["a2"], ["a3"], ["a4"]],
        },
        "map2": {
            "layout": """
  E 🪴 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩
 🟩 B  🟩 🚪 🟩 🟩 🟩 🚪 🟩 🟩 🟩 🚪 🟩  D 🟩
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
 ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 
 🪴 🟩 🟩 ⛔ 🥤 🪴 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
 🟩 O  🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩
 🟩 🟩 🟩 ⛔ 🥤 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
 ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🪴 🟩 🟩 ⛔ 🟩 🟩 🟩
 🟩 A  🟩 ⛔ 🟩 🪴 🟩 🚪 🟩 🟩 🟩 ⛔ 🪴 🟩 🟩
 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ ✉️ 🟩 🪴 ⛔ 🟩 🟩 🟩
 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ 🚪 ⛔
 🟩 🟩 🟩 🚪 🪴 🪴 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
 🟩 🟩 🟩 ⛔ C  🟩 🟩 🚪 🟩 🟩 🟩 🚪 🟩 🟩 🟩
 🪴 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🪴 🪴
 """,
            "grid_size": (12, 12),
            "position_map": lambda coordinates, goals: {
                coordinates["coffee"][0],
                coordinates["coffee"][1],
                coordinates["letter"][0],
                goals["A"],
                goals["B"],
                goals["C"],
                goals["D"],
                goals["E"],
                goals["O"],
            },
            "max_time": None,
            "agents": [
                {
                    "algorithm": "QRMAX",
                    "name": "a3",
                    "position": (2, 7),
                    "events": None,  # [((8, 1), 0), ((12, 6), 10)],
                    "actions": [move_up, move_down, move_left, move_right],
                },
            ],
            "groups_priority": None,  # [["a1"], ["a2"], ["a3"], ["a4"]],
        },
        "map3": {
            "layout": """
🟩 🪴 E  🚪 🟩 🟩 🪴 🚪 🪴 🪴 🪴 🚪 🪴 🟩 🟩 🚪 🟩 🪴 🟩
🟩 A  🪴 🚪 🟩 🟩 🟩 🚪 🟩 🟩 🪴 🚪 🟩 🟩 🟩 ⛔ 🟩 B  🟩
🟩 🟩 🟩 🚪 🟩 🟩 🟩 🚪 🟩 🟩 🟩 🚪 🟩 🟩 🪴 ⛔ 🟩 🟩 🟩
⛔ 🚪 ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ 🚪 🚪 ⛔
🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
🟩 🪴 🟩 ⛔ ✉️ 🟩 🟩 ⛔ 🟩 🟩 🥤 ⛔ 🟩 🪴 🪴 ⛔ 🟩 🪴 🟩
🟩 🟩 🟩 ⛔ 🪴 🪴 🪴 🚪 🟩 🪴 🪴 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
⛔ 🚪 ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ 🚪 ⛔
🪴 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩
🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩
🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ 🚪 🚪 ⛔ ⛔ ⛔ 🚪 ⛔
🟩 🟩 🪴 🚪 🪴 🟩 🥤 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🪴 🚪 🟩 🟩 🟩
🟩 D  🟩 ⛔ 🪴 🟩 🟩 🚪 🟩 🟩 🟩 🚪 🟩 🟩 🪴 ⛔ 🪴 🟩 🟩
🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
🚪 ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔
🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
🟩 O  🟩 🚪 🟩 🪴 🟩 🚪 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 C
🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩
""",
            "grid_size": (15, 15),
            "position_map": lambda coordinates, goals: {
                coordinates["coffee"][0],
                coordinates["coffee"][1],
                coordinates["letter"][0],
                goals["A"],
                goals["B"],
                goals["C"],
                goals["D"],
                goals["E"],
                goals["O"],
            },
            "max_time": None,
            "agents": [
                {
                    "algorithm": "RMAX",
                    "name": "a3",
                    "position": (2, 7),
                    "events": None,  # [((8, 1), 0), ((12, 6), 10)],
                    "actions": [move_up, move_down, move_left, move_right],
                },
            ],
            "groups_priority": None,  # [["a1"], ["a2"], ["a3"], ["a4"]],
        },
        "map4": {
            "layout": """
🟩 🪴 E  🚪 🟩 🟩 🪴 🚪 🪴 🪴 🪴 🚪 🪴 🟩 🟩 🚪 🟩 🪴 🥤
🟩 A  🟩 🚪 🟩 🟩 🟩 🚪 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ B  🟩 🟩
🟩 🟩 🟩 🚪 🟩 🟩 🟩 🚪 🟩 🟩 🟩 🚪 🟩 🟩 🪴 ⛔ 🪴 🟩 🟩
⛔ 🚪 ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ 🚪 🚪 ⛔
🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
🟩 🪴 🟩 ⛔ ✉️ 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🪴 ⛔ 🪴 🪴 🟩
🟩 🟩 🟩 ⛔ 🪴 🪴 🪴 🚪 🟩 🪴 🪴 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
⛔ 🚪 ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ 🚪 ⛔
🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩
🟩 🪴 🟩 ⛔ 🟩 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🟩 🪴 🟩
🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ 🚪 🚪 ⛔ ⛔ ⛔ 🚪 ⛔
🟩 🟩 🪴 🚪 🪴 🟩 🥤 ⛔ 🟩 🟩 🪴 ⛔ 🟩 🟩 🪴 🚪 🟩 🟩 🟩
🟩 D  🟩 ⛔ 🪴 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🪴 ⛔ 🪴 🟩 🟩
🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
🚪 ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔
🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
🟩 O  🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🟩 🪴 C
🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩
""",
            "grid_size": (15, 15),
            "position_map": lambda coordinates, goals: {
                coordinates["coffee"][0],
                coordinates["coffee"][1],
                coordinates["letter"][0],
                goals["A"],
                goals["B"],
                goals["C"],
                goals["D"],
                goals["E"],
                goals["O"],
            },
            "max_time": None,
            "agents": [
                {
                    "algorithm": "RMAX",
                    "name": "a3",
                    "position": (2, 7),
                    "events": None,  # [((8, 1), 0), ((12, 6), 10)],
                    "actions": [move_up, move_down, move_left, move_right],
                },
            ],
            "groups_priority": None,  # [["a1"], ["a2"], ["a3"], ["a4"]],
        },
    },
    # Global settings apply across all experiments and maps.
    "global_settings": {
        "max_step": 1e7,  # Maximum number of steps before the experiment is terminated.
        "max_time": None,  # No global time limit.
        "stochastic_env": True,  # Set to True for stochastic environments.
        "plants_penalty": -100,
        "wall_penalty": 0,
        "terminate_on_plants": False,
        "terminate_hit_walls": False,
        "early_stop_on_success": False,  # If True, the experiment stops early if the agent reaches a 100% success rate.
        "num_test_optimal_policy": 100,
    },
    "heat_map": {
        "save_heatmap_during_training": True,  # Set to False to disable saving during training
        "heatmap_start_episode": 1000,  # Episode from which to start saving the heatmap
    },
    # Transfer learning settings, useful for reusing learned behaviors.
    "transfer_learning": {
        "transition_environment": None,  # Set to False to disable transfer learning.
        "load_for_first_group": None,  # Set to False if the TE should not be loaded for the first group of agents.
    },
    # Weights & Biases (wandb) integration settings for logging and visualization.
    "wandb": {
        "activate": True,  # Set to True to enable wandb logging.
        "project": "EAQvsALL",  # "ICML25_exp7" - "ICML25_no_save_VI" # qrmax_seed_analysis, "office_world_stoc",  # The project name in wandb.
        "entity": "alee8",  # The wandb entity (user or team).
    },
}

# Function to get experiment details based on the selected map and experiment.
# This function parses the map layout and sets up the corresponding state transitions.
def get_experiment_for_map(map_name, selected_experiment):
    """Returns the experiment configuration based on the selected map and experiment."""
    # Parsing the layout of the selected map to identify key coordinates and goals.
    layout = config["maps"][map_name]["layout"]
    coordinates, goals, office_walls = parse_office_world(layout)

    # Define specific experiments that can be run on this map.
    experiments = {
        "exp1": {
            "description": "Coffee to Office",
            "transitions": {
                ("state0", coordinates["coffee"][0]): ("state1", 0),
                ("state0", coordinates["coffee"][1]): ("state1", 0),
                ("state1", goals["O"]): ("state2", 1),
            },
            "positions": {
                coordinates["coffee"][0],
                coordinates["coffee"][1],
                goals["O"],
            },
        },
        "exp2": {
            "description": "E-mail to Office",
            "transitions": {
                ("state0", coordinates["letter"][0]): ("state1", 0),
                ("state1", goals["O"]): ("state2", 1),
            },
            "positions": {
                coordinates["letter"][0],
                goals["O"],
            },
        },
        "exp3": {
            "description": "Coffee + Email to Office",
            "transitions": {
                ("state0", coordinates["letter"][0]): ("state1", 0),
                ("state0", coordinates["coffee"][0]): ("state2", 0),
                ("state0", coordinates["coffee"][1]): ("state2", 0),
                ("state2", coordinates["letter"][0]): ("state3", 0),
                ("state1", coordinates["coffee"][0]): ("state3", 0),
                ("state1", coordinates["coffee"][1]): ("state3", 0),
                ("state3", goals["O"]): ("state4", 1),
            },
            "positions": {
                coordinates["letter"][0],
                coordinates["coffee"][0],
                coordinates["coffee"][1],
                goals["O"],
            },
        },
        "exp4": {
            "description": "A-B-C-D",
            "transitions": {
                ("state0", goals["A"]): ("state1", 0),
                ("state1", goals["B"]): ("state2", 0),
                ("state2", goals["C"]): ("state3", 0),
                ("state3", goals["D"]): ("state4", 1),
            },
            "positions": {
                goals["A"],
                goals["B"],
                goals["C"],
                goals["D"],
            },
        },
        "exp5": {
            "description": "A-B-C-D, then Coffee + Email to Office",
            "transitions": {
                ("state0", goals["A"]): ("state1", 0),
                ("state1", goals["B"]): ("state2", 0),
                ("state2", goals["C"]): ("state3", 0),
                ("state3", goals["D"]): ("state4", 0),
                ("state4", coordinates["coffee"][0]): ("state5", 0),
                ("state4", coordinates["coffee"][1]): ("state5", 0),
                ("state4", coordinates["letter"][0]): ("state6", 0),
                ("state6", coordinates["coffee"][0]): ("state7", 0),
                ("state6", coordinates["coffee"][1]): ("state7", 0),
                ("state5", coordinates["letter"][0]): ("state7", 0),
                ("state7", goals["O"]): ("state8", 1),
            },
            "positions": {
                goals["A"],
                goals["B"],
                goals["C"],
                goals["D"],
                coordinates["coffee"][0],
                coordinates["coffee"][1],
                coordinates["letter"][0],
                goals["O"],
            },
        },
        "exp6": {
            "description": "A-B-C-D-E then Coffee + Email to Office",
            "transitions": {
                ("state0", goals.get("A")): ("state1", 0),
                ("state1", goals.get("B")): ("state2", 0),  # Use .get to avoid KeyError
                ("state2", goals["C"]): ("state3", 0),
                ("state3", goals["D"]): ("state4", 0),
                ("state4", goals.get("E")): ("state5", 0),
                ("state5", coordinates["coffee"][0]): ("state6", 0),
                ("state5", coordinates["coffee"][1]): ("state6", 0),
                ("state5", coordinates["letter"][0]): ("state7", 0),
                ("state7", coordinates["coffee"][0]): ("state8", 0),
                ("state7", coordinates["coffee"][1]): ("state8", 0),
                ("state6", coordinates["letter"][0]): ("state8", 0),
                ("state8", goals["O"]): ("state9", 1),
            },
            "positions": {
                goals["A"],
                goals["B"],
                goals["C"],
                goals["D"],
                goals.get("E"),
                coordinates["coffee"][0],
                coordinates["coffee"][1],
                coordinates["letter"][0],
                goals["O"],
            },
        },
        "exp7": {
            "description": "Coffee to Office",
            "transitions": {
                ("state0", coordinates["coffee"][0]): ("state1", 0),  # più distante
                ("state0", coordinates["coffee"][1]): ("state2", 0),
                ("state1", goals["O"]): (
                    "state3",
                    1000,
                ),  ##########################1000 o 100
                ("state2", goals["O"]): ("state3", 1),
            },
            "positions": {
                coordinates["coffee"][0],
                coordinates["coffee"][1],
                goals["O"],
            },
        },
        "exp0": {
            "description": "Letter + Coffe to Office",
            "transitions": {
                ("state0", coordinates["letter"][0]): ("state1", 0),
                ("state1", coordinates["coffee"][0]): ("state2", 0),
                ("state2", goals["O"]): ("state3", 1),
            },
            "positions": {
                coordinates["letter"][0],
                coordinates["coffee"][0],
                goals["O"],
            },
        },
        "exp0_simply": {
            "description": "Letter",
            "transitions": {
                ("state0", coordinates["letter"][0]): ("state1", 1),
            },
            "positions": {
                coordinates["letter"][0],
            },
        },
    }

    return experiments.get(selected_experiment)


"""
Experiment Parameters:

Deterministic Case:
-------------------
RMAX:
- action_space_size = 4
- s_a_threshold = 1
- max_reward = 1.0
- gamma = 0.99
- epsilon_one = 0.99

QL and QRM:
- action_space_size = 4
- learning_rate = 1
- gamma = 0.9
- action_selection = "greedy"
- epsilon_start = 0.1
- epsilon_end = 0.1
- qtable_init = 2

QRMAX:
- action_space_size = 4
- gamma = 0.99
- nsamplesTE = 1  # Transition Environment - threshold to consider a transition (s, a) known in the environment
- nsamplesRE = 1  # Reward Environment - threshold to consider the reward associated with a pair (s, a) known in the environment
- nsamplesTQ = 1  # Transition for Q - threshold to consider a state transition of the Reward Machine automaton (q, s') known given a pair (s, a)
- nsamplesRQ = 1  # Reward for Q - threshold to consider the reward associated with a transition of the Reward Machine automaton (q, s', q')

Stochastic Case:
----------------
RMAX:
- action_space_size = 4
- s_a_threshold = 60
- max_reward = 1.0
- gamma = 0.99
- epsilon_one = 0.99

QL and QRM:
- action_space_size = 4
- learning_rate = 0.1
- gamma = 0.9
- action_selection = "greedy"
- epsilon_start = 0.1
- epsilon_end = 0.1
- qtable_init = 2

QRMAX:
- action_space_size = 4
- gamma = 0.99
- nsamplesTE = 39  # Transition Environment - threshold to consider a transition (s, a) known in the environment
- nsamplesRE = 1  # Reward Environment - threshold to consider the reward associated with a pair (s, a) known in the environment
- nsamplesTQ = 1  # Transition for Q - threshold to consider a state transition of the Reward Machine automaton (q, s') known given a pair (s, a)
- nsamplesRQ = 1  # Reward for Q - threshold to consider the reward associated with a transition of the Reward Machine automaton (q, s', q')
"""
