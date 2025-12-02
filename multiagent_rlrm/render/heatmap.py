import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib import gridspec


# Example configuration snippets:
# dim_griglia = (10, 10)  # Grid dimensions (rows, columns)
# num_stati_rm = 3        # Number of states in the Reward Machine

# Caricamento della Q-table
# data = np.load('q_tables.npz')
# q_table = data['q_table_a3']  # Sostituisci 'q_table_a3' con la chiave corretta se necessario


def generate_heatmaps(q_table, dim_griglia, num_stati_rm):
    """
    Plot heatmaps of the maximum Q-values and greedy actions for each Reward Machine state.

    Args:
        q_table (np.ndarray): Q-table shaped either (nS, nA) or (nS, nQ, nA).
        dim_griglia (tuple): Grid dimensions as (height, width).
        num_stati_rm (int): Number of Reward Machine states.

    Returns:
        matplotlib.figure.Figure: Figure containing one heatmap per RM state.
    """
    grid_height, grid_width = dim_griglia

    action_symbols = {0: "↓", 1: "↑", 2: "←", 3: "→"}

    # Standardize q_table shape to (nS, nQ, nA)
    if q_table.ndim == 2:
        # QRM and QL: q_table shape is (nS, nA)
        nS, nA = q_table.shape
        nQ = num_stati_rm  # Number of RM states
        expected_nS = grid_height * grid_width * nQ
        assert (
            nS == expected_nS
        ), "Mismatch between grid size * num_rm_states and number of states"

        # Reshape q_table to (grid_height * grid_width, nQ, nA)
        q_table = q_table.reshape((grid_height * grid_width, nQ, nA))

    elif q_table.ndim == 3:
        # QRMAX: q_table shape is (nS, nQ, nA)
        nS, nQ, nA = q_table.shape
        expected_nS = grid_height * grid_width
        assert nS == expected_nS, "Mismatch between grid size and number of states"

    else:
        print(f"Unsupported q_table shape: {q_table.shape}")
        return

    # Now process each RM state
    fig, axes = plt.subplots(1, nQ, figsize=(20, 6))

    if nQ == 1:
        axes = [axes]

    for idx, q in enumerate(range(nQ)):
        # Extract q_values for RM state q
        if q_table.ndim == 3:
            # q_table is already in (nS, nQ, nA)
            q_values = q_table[:, q, :]  # Shape: (nS, nA)
        else:
            # q_table has been reshaped to (nS, nQ, nA)
            q_values = q_table[:, q, :]  # Shape: (nS, nA)

        # Reshape q_values to grid dimensions
        q_values_reshaped = q_values.reshape((grid_height, grid_width, nA))

        # Compute max Q-values and optimal actions
        max_q_values = q_values_reshaped.max(axis=2)
        optimal_actions = q_values_reshaped.argmax(axis=2)

        # Plot the heatmap
        sns.heatmap(
            max_q_values,
            annot=np.vectorize(action_symbols.get)(optimal_actions),
            fmt="",
            cmap="coolwarm",
            ax=axes[idx],
        )
        axes[idx].set_title(f"RM State {q + 1}")
        axes[idx].set_xlabel("Column")
        axes[idx].set_ylabel("Row")

    plt.tight_layout()
    # Instead of plt.show(), return the figure
    return fig


def generate_heatmaps_with_walls(
    q_table, dim_griglia, num_stati_rm, walls=None, plants=None
):
    """
    Plot heatmaps with greedy actions while also overlaying walls and optional plant markers.

    Args:
        q_table (np.ndarray): Q-table shaped either (nS, nA) or (nS, nQ, nA).
        dim_griglia (tuple): Grid dimensions as (height, width).
        num_stati_rm (int): Number of Reward Machine states.
        walls (list, optional): List of wall segments defined as ((x1, y1), (x2, y2)).
        plants (list, optional): Iterable of (x, y) coordinates to highlight.

    Returns:
        matplotlib.figure.Figure: Figure containing one heatmap per RM state.
    """
    grid_height, grid_width = dim_griglia

    action_symbols = {0: "↓", 1: "↑", 2: "←", 3: "→"}

    # Standardize q_table shape to (nS, nQ, nA)
    if q_table.ndim == 2:
        nS, nA = q_table.shape
        nQ = num_stati_rm
        expected_nS = grid_height * grid_width * nQ
        assert (
            nS == expected_nS
        ), "Mismatch between grid size * num_rm_states and number of states"
        q_table = q_table.reshape((grid_height * grid_width, nQ, nA))
    elif q_table.ndim == 3:
        nS, nQ, nA = q_table.shape
        expected_nS = grid_height * grid_width
        assert nS == expected_nS, "Mismatch between grid size and number of states"
    else:
        print(f"Unsupported q_table shape: {q_table.shape}")
        return

    fig, axes = plt.subplots(1, nQ, figsize=(20, 6))

    if nQ == 1:
        axes = [axes]

    for idx, q in enumerate(range(nQ)):
        q_values = q_table[:, q, :]
        q_values_reshaped = q_values.reshape((grid_height, grid_width, nA))

        max_q_values = q_values_reshaped.max(axis=2)
        optimal_actions = q_values_reshaped.argmax(axis=2)

        # Plot the heatmap
        sns.heatmap(
            max_q_values,
            annot=np.vectorize(action_symbols.get)(optimal_actions),
            fmt="",
            cmap="coolwarm",
            ax=axes[idx],
            cbar=False,
            square=True,
            linewidths=0.5,
            linecolor="gray",
        )

        axes[idx].set_title(f"RM State {q + 1}")
        axes[idx].set_xlabel("Column")
        axes[idx].set_ylabel("Row")

        # Keep the default Y-axis orientation
        # axes[idx].invert_yaxis()

        # Draw the walls
        if walls is not None:
            cell_size = 1  # In the heatmap, each cell corresponds to 1 unit
            for ((x1, y1), (x2, y2)) in walls:
                if x1 == x2:  # Vertical wall
                    start_pos = (x1 * cell_size, min(y1, y2) * cell_size + cell_size)
                    end_pos = (
                        (x1 + 1) * cell_size,
                        min(y1, y2) * cell_size + cell_size,
                    )
                    axes[idx].plot(
                        [start_pos[0], end_pos[0]],
                        [start_pos[1], end_pos[1]],
                        "k-",
                        lw=2,
                    )
                elif y1 == y2:  # Horizontal wall
                    start_pos = (min(x1, x2) * cell_size + cell_size, y1 * cell_size)
                    end_pos = (
                        min(x1, x2) * cell_size + cell_size,
                        (y1 + 1) * cell_size,
                    )
                    axes[idx].plot(
                        [start_pos[0], end_pos[0]],
                        [start_pos[1], end_pos[1]],
                        "k-",
                        lw=2,
                    )

        # Draw the plants
        if plants is not None:
            for (p_x, p_y) in plants:
                axes[idx].scatter(
                    p_x + 0.5,
                    p_y + 0.5,
                    marker="X",
                    s=100,
                    color="green",
                    label="Plant",
                )
            # Add legend if plants are drawn
            # axes[idx].legend(loc='upper right')

    plt.tight_layout()
    return fig


def generate_heatmaps_time(q_table, dim_griglia, num_stati_rm, max_time):
    """
    Plot heatmaps for time-indexed Q-tables by averaging across the time dimension.

    Args:
        q_table (np.ndarray): Q-table shaped (height, width, time, rm_states, actions).
        dim_griglia (tuple): Grid dimensions as (height, width).
        num_stati_rm (int): Number of Reward Machine states.
        max_time (int): Maximum time dimension used in the Q-table.
    """
    # Compute mean Q-values per RM state and position, averaging over time
    reshaped_q_table = q_table.reshape((*dim_griglia, max_time, num_stati_rm, -1))
    mean_q_values = reshaped_q_table.mean(axis=2)  # Mean along time axis

    max_q_values = mean_q_values.max(axis=-1)  # Maximum Q-values
    optimal_actions = mean_q_values.argmax(axis=-1)  # Greedy action indices

    # Mappa dei codici delle azioni ai simboli corrispondenti
    action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    # Heatmap visualization
    fig, axes = plt.subplots(1, num_stati_rm, figsize=(20, 6))

    if num_stati_rm == 1:
        axes = [axes]  # Assicurati che axes sia sempre un array

    for i in range(num_stati_rm):
        sns.heatmap(
            max_q_values[:, :, i],
            annot=np.vectorize(action_symbols.get)(optimal_actions[:, :, i]),
            fmt="",
            cmap="coolwarm",
            ax=axes[i],
        )
        axes[i].set_title(f"RM State {i + 1}")
        axes[i].set_xlabel("Column")
        axes[i].set_ylabel("Row")

    plt.tight_layout()
    plt.show()


"""
# Assuming q_tables_dict contains Q-tables for all agents as shown in your code
for agent_name, q_table in data.items():
    generate_heatmaps(q_table, dim_griglia=(10, 10), num_stati_rm=3)  # Adjust RM_3 to the correct RM instance if necessary
"""


def generate_heatmaps_for_agents(
    agents, q_tables_data, grid_dims, walls=None, plants=None
):
    """
    Generates heatmaps for the Q-tables of each agent.

    Args:
        agents (list): List of agents.
        q_tables_data (dict): Dictionary containing Q-tables for each agent.
        grid_dims (tuple): Grid dimensions (grid_height, grid_width).
        walls (list): List of walls (optional).
        plants (list): List of plant positions (optional).
    """
    heatmap_figures = []
    for agent in agents:
        agent_name = agent.name
        q_table = q_tables_data.get(f"q_table_{agent_name}")
        if q_table is None:
            print(f"No Q-table data for agent {agent_name}")
            continue

        num_rm_states = agent.get_reward_machine().numbers_state()
        fig = generate_heatmaps_with_walls(
            q_table,
            dim_griglia=grid_dims,
            num_stati_rm=num_rm_states,
            walls=walls,
            plants=plants,
        )

        heatmap_figures.append((agent_name, fig))
    return heatmap_figures


def generate_heatmaps_for_agents_time(agents, q_tables_data, grid_dims, max_time):
    """
    Generate time-averaged heatmaps for each agent's Q-table.

    Args:
        agents (list): Agents whose Q-tables should be visualized.
        q_tables_data (np.lib.npyio.NpzFile): Loaded Q-table data keyed by agent.
        grid_dims (tuple): Grid dimensions, e.g., (width, height).
        max_time (int): Number of time steps encoded in the Q-table.
    """
    for agent in agents:
        agent_name = agent.name
        q_table = q_tables_data[
            f"q_table_{agent_name}"
        ]  # Recupera la Q-table dell'agente corrente
        num_rm_states = (
            agent.get_reward_machine().numbers_state()
        )  # Ottiene il numero di stati RM dinamicamente

        # Chiama la funzione generate_heatmaps per ogni agente
        # Assumendo che generate_heatmaps accetti i seguenti parametri: q_table, dimensioni griglia, e numero stati RM
        generate_heatmaps_time(
            q_table,
            dim_griglia=grid_dims,
            num_stati_rm=num_rm_states,
            max_time=max_time,
        )


def extract_q_tables(agents):
    """
    Collect Q-table data from each agent's learning algorithm.

    Args:
        agents (list): Agents whose Q-tables should be gathered.

    Returns:
        dict: Mapping of agent names to their Q-table arrays.
    """
    q_tables_data = {}
    for agent in agents:
        algorithm = agent.get_learning_algorithm()
        if hasattr(algorithm, "q_table"):
            q_table = algorithm.q_table  # For QRM and QL
        elif hasattr(algorithm, "Q"):
            q_table = algorithm.Q  # For QRMAX and RMAX
        else:
            print(f"No Q-table found for agent {agent.name}")
            continue
        q_tables_data[f"q_table_{agent.name}"] = q_table
    return q_tables_data


def generate_value_policy_heatmap(
    V, policy, grid_dims, rm, walls=None, plants=None, goals=None, coordinates=None
):
    """
    Render value and policy heatmaps for each Reward Machine state.

    Args:
        V (np.ndarray): Flattened value function aligned with (grid, rm state) ordering.
        policy (np.ndarray): Greedy action indices for each combined state.
        grid_dims (tuple): Grid dimensions as (height, width).
        rm (RewardMachine): Reward machine providing state metadata.
        walls (list, optional): Wall segments to overlay.
        plants (list, optional): Plant coordinates to overlay.
        goals (dict, optional): Mapping of goal labels to coordinates.
        coordinates (dict, optional): Mapping of item types to coordinate sets.

    Returns:
        matplotlib.figure.Figure: Figure showing value/policy per RM state.
    """
    grid_height, grid_width = grid_dims
    num_states = len(V)
    num_rm_states = rm.numbers_state()
    expected_states = grid_height * grid_width * num_rm_states
    assert num_states == expected_states, f"Mismatch in number of states."

    # Map RM state indices back to names
    index_to_state = {v: k for k, v in rm.state_indices.items()}
    rm_states_list = [index_to_state[i] for i in range(num_rm_states)]

    action_symbols = {0: "↓", 1: "↑", 2: "←", 3: "→"}

    fig, axes = plt.subplots(1, num_rm_states, figsize=(8 * num_rm_states, 8))
    if num_rm_states == 1:
        axes = [axes]

    transitions = rm.get_transitions  # { (from_state, event): (to_state, reward) }

    pos_to_letter = {}
    if goals is not None:
        for letter, pos in goals.items():
            pos_to_letter[pos] = letter

    coffee_positions = set(coordinates.get("coffee", [])) if coordinates else set()
    letter_positions = set(coordinates.get("letter", [])) if coordinates else set()

    # Load coffee and letter icons as RGBA arrays
    img_path = os.path.join(os.path.dirname(__file__))

    coffee_image = plt.imread(
        f"{img_path}/img/coffee.png"
    )  # Ensure the path is correct
    letter_image = plt.imread(f"{img_path}/img/email.png")  # Ensure the path is correct

    for idx, rm_state_name in enumerate(rm_states_list):
        q_rm = rm.state_indices[rm_state_name]
        values_2d = np.zeros((grid_height, grid_width))
        policy_2d = np.zeros((grid_height, grid_width), dtype=int)

        for y in range(grid_height):
            for x in range(grid_width):
                s = (y * grid_width + x) * num_rm_states + q_rm
                values_2d[y, x] = V[s]
                policy_2d[y, x] = policy[s]

        annot = np.empty(policy_2d.shape, dtype=object)
        for yy in range(grid_height):
            for xx in range(grid_width):
                annot[yy, xx] = action_symbols.get(policy_2d[yy, xx], "?")

        ax = axes[idx]

        sns.heatmap(
            values_2d,
            annot=annot,
            fmt="",
            cmap="coolwarm",
            ax=ax,
            cbar=False,
            annot_kws={"size": 16, "weight": "bold"},
        )
        ax.set_title(f"RM State {rm_state_name} (index={q_rm})")
        ax.set_xlabel("X (column)")
        ax.set_ylabel("Y (row)")

        # Keep square cells
        ax.set_aspect("equal")

        # Draw plants as green X markers (optional)
        if plants is not None:
            for (p_x, p_y) in plants:
                ax.scatter(p_x + 0.5, p_y + 0.5, marker="X", s=200, color="green")

        # Draw walls
        if walls is not None:
            for ((x1, y1), (x2, y2)) in walls:
                if x1 == x2:
                    row = min(y1, y2)
                    ax.plot([x1, x1 + 1], [row + 1, row + 1], "k-", lw=2)
                elif y1 == y2:
                    col = min(x1, x2)
                    ax.plot([col + 1, col + 1], [y1, y1 + 1], "k-", lw=2)

        # Show transitions with priorities:
        # 1. Draw the goal letter when available.
        # 2. If (g_x, g_y) in coffee_positions -> show coffee image.
        # 3. If (g_x, g_y) in letter_positions -> show letter image.
        # 4. Otherwise render a star/diamond with the target state name.
        for (from_st, event), (to_st, rew) in transitions.items():
            if from_st == rm_state_name:
                g_x, g_y = event
                letter = pos_to_letter.get((g_x, g_y), "")

                if letter:
                    # Goal location with a letter marker
                    if rew > 0:
                        # ax.scatter(g_x+0.5, g_y+0.5, marker='*', s=350, color='gold', edgecolors='black', linewidths=1)
                        ax.text(
                            g_x + 0.5,
                            g_y + 0.5,
                            letter,
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=14,
                            fontweight="bold",
                        )
                    else:
                        ax.scatter(
                            g_x + 0.5,
                            g_y + 0.5,
                            marker="D",
                            s=300,
                            color="darkgreen",
                            edgecolors="black",
                            linewidths=1,
                        )
                        ax.text(
                            g_x + 0.5,
                            g_y + 0.5,
                            letter,
                            ha="center",
                            va="center",
                            color="white",
                            fontsize=10,
                            fontweight="bold",
                        )
                else:
                    # No letter provided for this goal
                    if (g_x, g_y) in coffee_positions:
                        # Show coffee icon, slightly reduced to fit cell
                        ax.imshow(
                            coffee_image,
                            extent=[g_x + 0.1, g_x + 0.9, g_y + 0.9, g_y + 0.1],
                            origin="upper",
                            zorder=10,
                        )
                    elif (g_x, g_y) in letter_positions:
                        # Show letter icon
                        ax.imshow(
                            letter_image,
                            extent=[g_x + 0.1, g_x + 0.9, g_y + 0.9, g_y + 0.1],
                            origin="upper",
                            zorder=10,
                        )
                    else:
                        # Fallback rendering
                        if rew > 0:
                            # ax.scatter(g_x + 0.5, g_y + 0.5, marker='*', s=300, color='gold', edgecolors='black', linewidths=1)
                            ax.text(
                                g_x + 0.5,
                                g_y + 0.5,
                                to_st,
                                ha="center",
                                va="center",
                                color="black",
                                fontsize=14,
                                fontweight="bold",
                            )
                        else:
                            ax.scatter(
                                g_x + 0.5,
                                g_y + 0.5,
                                marker="D",
                                s=300,
                                color="darkgreen",
                                edgecolors="black",
                                linewidths=1,
                            )
                            ax.text(
                                g_x + 0.5,
                                g_y + 0.5,
                                to_st[-1] if to_st else "",
                                ha="center",
                                va="center",
                                color="white",
                                fontsize=10,
                            )

    plt.tight_layout()
    return fig


def generate_value_heatmap(V, grid_dims, rm, walls=None, plants=None, goals=None):
    """
    Generates heatmaps for the value function (V) for each Reward Machine (RM) state.

    Parameters:
    - V: np.ndarray, the value function, a flat array of state values.
    - grid_dims: tuple, (grid_height, grid_width) dimensions of the grid.
    - rm: Reward Machine object, containing RM state indices.
    - walls: list of tuples, coordinates defining walls (optional).
    - plants: list of tuples, coordinates of plants to highlight (optional).
    - goals: dict, mapping of goal positions to their labels (optional).

    Returns:
    - fig: Matplotlib figure object containing the heatmaps.
    """
    grid_height, grid_width = grid_dims
    num_states = len(V)
    num_rm_states = rm.numbers_state()
    expected_states = grid_height * grid_width * num_rm_states

    # Ensure value function matches expected states
    assert (
        num_states == expected_states
    ), "Mismatch between value function and grid size."

    # Map RM state indices to state names
    index_to_state = {v: k for k, v in rm.state_indices.items()}
    rm_states_list = [index_to_state[i] for i in range(num_rm_states)]

    # Create a figure with GridSpec for better layout control
    fig = plt.figure(figsize=(8 * num_rm_states, 8))
    spec = gridspec.GridSpec(1, num_rm_states, wspace=0.3)

    # Generate heatmaps for each RM state
    for idx, rm_state_name in enumerate(rm_states_list):
        q_rm = rm.state_indices[rm_state_name]
        values_2d = np.zeros((grid_height, grid_width))

        for y in range(grid_height):
            for x in range(grid_width):
                s = (y * grid_width + x) * num_rm_states + q_rm
                values_2d[y, x] = V[s]

        ax = plt.subplot(spec[idx])

        # Plot heatmap without colorbar
        sns.heatmap(
            values_2d,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax,
            cbar=False,
            square=True,
            annot_kws={"size": 8},
        )
        ax.set_title(f"RM State {rm_state_name} (index={q_rm})", fontsize=12)
        ax.set_xlabel("X (column)", fontsize=10)
        ax.set_ylabel("Y (row)", fontsize=10)

        # Add plants as markers
        if plants:
            for (p_x, p_y) in plants:
                ax.scatter(p_x + 0.5, p_y + 0.5, marker="X", s=100, color="green")

        # Add walls as lines
        if walls:
            for ((x1, y1), (x2, y2)) in walls:
                if x1 == x2:  # Vertical wall
                    row = min(y1, y2)
                    ax.plot([x1, x1 + 1], [row + 1, row + 1], "k-", lw=1)
                elif y1 == y2:  # Horizontal wall
                    col = min(x1, x2)
                    ax.plot([col + 1, col + 1], [y1, y1 + 1], "k-", lw=1)

    # Add a global title
    fig.suptitle(
        "Value Function Heatmap Across RM States", fontsize=16, fontweight="bold"
    )
    # plt.tight_layout(pad=1.0)
    return fig
