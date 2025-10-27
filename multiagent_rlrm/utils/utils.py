import numpy as np
import string


def encode_state(agent, state, state_reward_machine):
    """
    Codifica lo stato corrente e lo stato della Reward Machine in un singolo indice intero.

    :param agent: L'agente per il quale codificare lo stato.
    :param state: Lo stato corrente dell'agente.
    :return: Indice intero che rappresenta lo stato codificato.
    """

    # Estrai le informazioni necessarie da 'agent' e 'state'

    RM_agent = agent.get_reward_machine()
    num_rm_states = RM_agent.numbers_state()
    pos_x, pos_y = state["pos_x"], state["pos_y"]
    rm_state_index = RM_agent.get_state_index(state_reward_machine)
    max_x_value, max_y_value = agent.ma_problem.grid_width, agent.ma_problem.grid_height

    # Calcola l'indice basato sulla posizione
    pos_index = pos_y * max_x_value + pos_x

    # Codifica lo stato combinando la posizione e lo stato della Reward Machine
    encoded_state = pos_index * num_rm_states + rm_state_index

    # Controlla che l'indice codificato non superi le dimensioni totali dello spazio degli stati
    if encoded_state >= max_x_value * max_y_value * num_rm_states:
        raise ValueError(
            "Indice di stato codificato supera la dimensione dello spazio degli stati",
            encoded_state,
            ">=",
            max_x_value * max_y_value * num_rm_states,
        )

    return encoded_state


def encode_state_with_time(agent, state, state_reward_machine):
    """
    Codifica lo stato corrente, lo stato della Reward Machine e il timestamp in un singolo indice intero.

    :param agent: L'agente per il quale codificare lo stato.
    :param state: Lo stato corrente dell'agente, inclusi pos_x, pos_y e timestamp.
    :return: Indice intero che rappresenta lo stato codificato.
    """

    # Estrai le informazioni necessarie da 'agent' e 'state'
    RM_agent = agent.get_reward_machine()
    num_rm_states = RM_agent.numbers_state()
    pos_x, pos_y, time_index = state["pos_x"], state["pos_y"], state["timestamp"]
    rm_state_index = RM_agent.get_state_index(state_reward_machine)
    max_x_value, max_y_value = agent.ma_problem.grid_width, agent.ma_problem.grid_height
    max_time_value = (
        agent.ma_problem.max_time
    )  # Assicurati che max_time sia definito nell'ambiente/problem

    # Calcola l'indice basato sulla posizione e sul tempo
    pos_index = pos_y * max_x_value + pos_x
    time_component = time_index  # Potresti voler scalare o modificare questa componente in base alle necessità

    # Codifica lo stato combinando la posizione, il tempo e lo stato della Reward Machine
    # Ora la formula tiene conto della dimensione temporale oltre alle dimensioni spaziali e allo stato della RM
    encoded_state = (
        pos_index * max_time_value + time_component
    ) * num_rm_states + rm_state_index

    # Controlla che l'indice codificato non superi le dimensioni totali dello spazio degli stati
    total_states = max_x_value * max_y_value * max_time_value * num_rm_states
    if encoded_state >= total_states:
        raise ValueError(
            "Indice di stato codificato supera la dimensione dello spazio degli stati",
            encoded_state,
            ">=",
            total_states,
        )

    return encoded_state


def encode_state_time(agent, state, state_reward_machine):
    """
    Codifica lo stato corrente, lo stato della Reward Machine, e il timestap in un singolo indice intero.

    :param agent: L'agente per il quale codificare lo stato.
    :param state: Lo stato corrente dell'agente.
    :param state_reward_machine: Stato corrente della Reward Machine.
    :param timestep: Timestap corrente, un intero da 0 a 50.
    :return: Indice intero che rappresenta lo stato codificato.
    """
    # Estrai le informazioni necessarie da 'agent' e 'state'
    RM_agent = agent.get_reward_machine()
    num_rm_states = RM_agent.numbers_state()
    pos_x, pos_y, timestep = state["pos_x"], state["pos_y"], state["timestep"]
    rm_state_index = RM_agent.get_state_index(state_reward_machine)
    max_x_value, max_y_value = agent.ma_problem.grid_width, agent.ma_problem.grid_height
    num_timesteps = (
        agent.ma_problem.max_time
    )  # Timesteps vanno da 0 a 50, quindi abbiamo 51 timesteps possibili

    # Calcola l'indice basato sulla posizione
    pos_index = pos_y * max_x_value + pos_x

    # Estendi la codifica per includere il tempo
    base_index = pos_index * num_rm_states + rm_state_index
    encoded_state = base_index * num_timesteps + timestep
    # encoded_state = np.int16(encoded_state)

    # Calcola il numero totale possibile di stati
    total_states = max_x_value * max_y_value * num_rm_states * num_timesteps

    # print(base_index, num_timesteps, timestep, "total states:", total_states, "state:", state)

    # Controlla che l'indice codificato non superi le dimensioni totali dello spazio degli stati
    if encoded_state >= total_states:
        raise ValueError(
            "Indice di stato codificato supera la dimensione dello spazio degli stati",
            encoded_state,
            ">=",
            total_states,
        )

    return encoded_state


def encode_environment_state(agent, state):
    max_x_value = agent.ma_problem.grid_width
    pos_x, pos_y = state["pos_x"], state["pos_y"]
    encoded_state = pos_y * max_x_value + pos_x
    return encoded_state


def encode_environment_state_time(agent, state):
    max_x_value = agent.ma_problem.grid_width
    pos_x, pos_y, timestep = state["pos_x"], state["pos_y"], state["timestep"]
    num_timesteps = agent.ma_problem.max_time
    pos_index = pos_y * max_x_value + pos_x
    encoded_state = pos_index * num_timesteps + timestep
    return encoded_state


"""def encode_environment_state(agent, state):
    max_x_value = agent.ma_problem.grid_width
    max_time_value = agent.ma_problem.max_time
    pos_x, pos_y, time_index = state["pos_x"], state["pos_y"], state['timestamp']

    pos_index = pos_y * max_x_value + pos_x
    #time_index = timestamp

    encoded_state = pos_index * max_time_value + time_index
    return encoded_state"""


def encode_reward_machine_state(agent, q):
    # Assumi che RM_agent mantenga un elenco ordinato degli stati o una mappatura da stati a indici
    RM_agent = agent.get_reward_machine()
    encoded_q = RM_agent.get_state_index(
        q
    )  # Supponendo che esista una funzione come questa
    return encoded_q


def parse_map_string(map_string):
    holes = []
    goals = {}
    for y, row in enumerate(map_string.strip().split("\n")):
        for x, cell in enumerate(row.strip()):
            if cell == "H":
                holes.append((x, y))
            elif cell.isdigit():  # Controlla se il carattere è un numero
                goals[int(cell)] = (x, y)  # Memorizza il goal con il numero come chiave
    return holes, goals


import textwrap


def parse_map_emoji(map_string):
    """
    Parsifica una mappa emoji dove:
      - '⛔' sono buche
      - lettere o cifre sono goal
      - ogni altro carattere (ad es. 🟩) è un semplice pavimento
    Gli spazi (inclusa indentazione) vengono completamente ignorati.

    Ritorna:
      holes: list of (x, y)
      goals: dict {char: (x, y)}
      dims: (width, height)
    """
    # 1) rimuovo l’indentazione comune e le righe vuote iniziali/finali
    map_string = textwrap.dedent(map_string).strip()
    lines = map_string.splitlines()

    holes = []
    goals = {}

    for y, raw_row in enumerate(lines):
        # 2) ignoro *tutti* gli spazi
        cells = [c for c in raw_row if c != " "]
        for x, cell in enumerate(cells):
            if cell == "⛔":
                holes.append((x, y))
            elif cell.isdigit() or cell.isalpha():
                goals[cell] = (x, y)
        # gli altri (🟩 ecc.) li scarto

    # 3) dimensioni: larghezza massima (conteggio di celle per riga), altezza = numero di righe
    width = max(len([c for c in row if c != " "]) for row in lines)
    height = len(lines)

    return holes, goals, (width, height)


def parse_office_world(office_world):
    symbol_mapping = {"🟩": "empty_cell", "🪴": "plant", "🥤": "coffee", "✉️": "letter"}
    # Parse the office world into a list of lists, ignoring ⛔ and 🚪
    office_lines = [
        line.replace("⛔", "").replace("🚪", "").strip().split()
        for line in office_world.strip().split("\n")
    ]
    filtered_list_of_lists = [lst for lst in office_lines if lst]
    goals = {
        char: [] for char in string.ascii_uppercase + string.digits
    }  # Dictionary for alphabetic and numeric goals

    # Initialize dictionaries to hold the coordinates for each symbol
    coordinates = {"plant": [], "coffee": [], "letter": [], "empty_cell": []}

    # Iterate through the office lines and collect the coordinates

    # Iterate through the office lines and collect the coordinates
    for y, row in enumerate(filtered_list_of_lists):
        for x, cell in enumerate(row):
            if cell in symbol_mapping:
                coordinates[symbol_mapping[cell]].append((x, y))
            elif cell in string.ascii_uppercase + string.digits:
                goals[cell].append((x, y))

    # Filter out empty goal lists
    goals = {k: v[0] for k, v in goals.items() if v}

    walls = find_disconnected_pairs(office_world)

    return (
        coordinates,
        goals,
        walls,
    )


def parse_office_grid(office_world):
    # Parsing the grid into a list of lists
    grid = [line.strip().split() for line in office_world.strip().split("\n")]
    return grid


"""def find_disconnected_pairs(office_world):
    grid = parse_office_grid(office_world)
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    disconnected_pairs = []

    y_offsets = [0] * rows
    y_adjusted = 0
    for y in range(rows):
        if all(cell == "⛔" or cell == "🚪" for cell in grid[y]):
            y_offsets[y] = 1
        else:
            y_offsets[y] = y_adjusted
            y_adjusted += 1

    x_offsets = [0] * cols
    x_adjusted = 0
    for x in range(cols):
        if all(grid[y][x] == "⛔" or grid[y][x] == "🚪" for y in range(rows)):
            x_offsets[x] = 1
        else:
            x_offsets[x] = x_adjusted
            x_adjusted += 1

    for y in range(rows):
        for x in range(cols):
            if grid[y][x] == "⛔":
                continue
            if (
                x < cols - 1
                and grid[y][x + 1] == "⛔"
                and x + 2 < cols
                and grid[y][x + 2] != "⛔"
            ):
                disconnected_pairs.append(
                    ((x_offsets[x], y_offsets[y]), (x_offsets[x + 2], y_offsets[y]))
                )
            if (
                y < rows - 1
                and grid[y + 1][x] == "⛔"
                and y + 2 < rows
                and grid[y + 2][x] != "⛔"
            ):
                disconnected_pairs.append(
                    ((x_offsets[x], y_offsets[y]), (x_offsets[x], y_offsets[y + 2]))
                )

    return disconnected_pairs"""


def find_disconnected_pairs(office_world):
    grid = parse_office_grid(office_world)
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    disconnected_pairs = []

    y_offsets = [0] * rows
    y_adjusted = 0
    for y in range(rows):
        if all(cell == "⛔" or cell == "🚪" for cell in grid[y]):
            y_offsets[y] = 1
        else:
            y_offsets[y] = y_adjusted
            y_adjusted += 1

    x_offsets = [0] * cols
    x_adjusted = 0
    for x in range(cols):
        if all(grid[y][x] == "⛔" or grid[y][x] == "🚪" for y in range(rows)):
            x_offsets[x] = 1
        else:
            x_offsets[x] = x_adjusted
            x_adjusted += 1

    for y in range(rows):
        for x in range(cols):
            if grid[y][x] == "⛔" or grid[y][x] == "🚪":
                continue
            # Debug: Print current cell being checked
            # print(f"Checking cell ({x}, {y}): {grid[y][x]}")
            # Check horizontal pairs
            if (
                x < cols - 2  # Check two cells ahead
                and grid[y][x + 1] == "⛔"
                and grid[y][x + 2] != "⛔"
            ):
                disconnected_pairs.append(
                    ((x_offsets[x], y_offsets[y]), (x_offsets[x + 2], y_offsets[y]))
                )
                # Debug: Print added horizontal pair
                # print(
                #    f"Added pair horizontal: (({x_offsets[x]}, {y_offsets[y]}), ({x_offsets[x + 2]}, {y_offsets[y]}))"
                # )
            # Check vertical pairs
            if (
                y < rows - 2  # Check two cells below
                and grid[y + 1][x] == "⛔"
                and grid[y + 2][x] != "⛔"
                and not (
                    x == 1 and y == 3 and grid[1][4] == "🪴"
                )  # Avoid erroneous pair
            ):
                disconnected_pairs.append(
                    ((x_offsets[x], y_offsets[y]), (x_offsets[x], y_offsets[y + 2]))
                )
                # Debug: Print added vertical pair
                # print(
                #    f"Added pair vertical: (({x_offsets[x]}, {y_offsets[y]}), ({x_offsets[x]}, {y_offsets[y + 2]}))"
                # )

    # Debug: Print final disconnected pairs
    # print("Final disconnected pairs:", disconnected_pairs)

    return disconnected_pairs


def generate_transitions(obstacles, goals, max_time):
    states = ["state_" + str(i) for i in range(max_time + 1)]
    states += [f"state_reached_1_{i}" for i in range(max_time + 1)]
    states += [f"state_reached_2_{i}" for i in range(max_time + 1)]
    states.append("free_play")

    transitions = {}

    # Aggiungi transizioni per collisioni
    for state in states:
        for x, y, t in obstacles:
            transitions[(state, f"collision_{x}_{y}_{t}")] = (state, -20)

    # Transizioni per avanzamento del tempo
    for i in range(max_time):
        transitions[(f"state_{i}", f"timestep_{i + 1}")] = (f"state_{i + 1}", 0)
        transitions[(f"state_reached_1_{i}", f"timestep_{i + 1}")] = (
            f"state_reached_1_{i + 1}",
            0,
        )
        transitions[(f"state_reached_2_{i}", f"timestep_{i + 1}")] = (
            f"state_reached_2_{i + 1}",
            0,
        )

    # Transizioni al free_play dopo il timestep 25
    transitions[(f"state_{max_time}", "timestep_{max_time+1}")] = ("free_play", 0)
    transitions[(f"state_reached_1_{max_time}", "timestep_{max_time+1}")] = (
        "free_play",
        0,
    )
    transitions[(f"state_reached_2_{max_time}", "timestep_{max_time+1}")] = (
        "free_play",
        0,
    )

    # Transizioni per gli obiettivi
    for state in states[: max_time + 1] + [
        f"state_reached_1_{i}" for i in range(max_time + 1)
    ]:
        transitions[(state, "(2, 1)")] = (f"state_reached_1_{0}", 10)
    for state in [f"state_reached_1_{i}" for i in range(max_time + 1)]:
        transitions[(state, "(12, 2)")] = (f"state_reached_2_{0}", 20)

    # Cicli all'interno dello stato free_play
    transitions[("free_play", "continue")] = ("free_play", 0)

    return transitions
