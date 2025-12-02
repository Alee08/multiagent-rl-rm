from typing import Optional, List, Union, Iterable
from collections import deque


class RewardMachine:
    def __init__(self, transitions, event_detector):
        """
        Reward Machine that encodes a reward structure as an automaton.

        :param transitions: Dict of the form {(current_state, event): (new_state, reward)}.
        :param event_detector: Object responsible for detecting events from environment states.
        """
        self.transitions = transitions  # {(current_state, event): (new_state, reward)}
        self.initial_state = self._get_start_state()  # Store the initial state
        self.current_state = self.initial_state
        self.state_indices = self._generate_state_indices()
        self.event_detector = event_detector
        self.potentials = None  # Store potentials for reward shaping

    def _generate_state_indices(self):
        """
        Collect all unique states (both source and target) from transitions
        and assign each of them a unique index. The initial state is always
        mapped to index 0.
        """
        # Collect all unique states (both source and target) from transitions
        unique_states = set()
        for (from_state, _), (to_state, _) in self.transitions.items():
            unique_states.add(from_state)
            unique_states.add(to_state)

        # Ensure the initial state is included and mapped to zero
        unique_states.add(self.current_state)
        sorted_states = sorted(unique_states)
        sorted_states.remove(self.current_state)
        sorted_states.insert(0, self.current_state)

        # Assign a unique index to each state
        return {state: i for i, state in enumerate(sorted_states)}

    def get_state_index(self, rm_state):
        """Return the integer index corresponding to the given Reward Machine state."""
        return self.state_indices[rm_state]

    def step(self, current_state):
        """
        Detect the current event, perform the corresponding state transition,
        and return the associated reward.

        :param current_state: Environment state used by the event detector.
        :return: Reward produced by the transition (0 if no transition is triggered).
        """
        event = self.event_detector.detect_event(current_state)
        # print(f"Detected event: {event} for current state: {current_state}")
        if (self.current_state, event) in self.transitions:
            new_state, reward = self.transitions[(self.current_state, event)]
            self.current_state = new_state
            return reward
        return 0

    def get_reward(self, event):
        """
        Return the reward associated with a specific event and also
        update the current state if the transition exists.

        :param event: Event label to evaluate.
        :return: Reward value (0 if there is no matching transition).
        """
        if (self.current_state, event) in self.transitions:
            new_state, reward = self.transitions[(self.current_state, event)]
            self.current_state = new_state
            return reward

        return 0

    def get_reward_for_non_current_state(self, state_rm, event):
        """
        Return the (next_state, reward) for a given RM state and event,
        without using or modifying self.current_state.

        :param state_rm: Reward Machine state (not necessarily the current one).
        :param event: Event label (can be converted to hashable if needed).
        :return: (next_state, reward) if transition exists, otherwise (None, 0).
        """
        # Convert event to a hashable type if needed
        if isinstance(event, list):
            print("It's a list!")
            breakpoint()
            event = tuple(event)

        if (state_rm, event) in self.transitions:
            new_state, reward = self.transitions[(state_rm, event)]
            return new_state, reward
        else:
            return None, 0

    def get_all_states(self):
        """
        Return a list of all states appearing in transitions (source or target),
        preserving their order of first appearance.
        """
        seen_states = set()
        all_states = []

        for (from_state, _), (to_state, _) in self.transitions.items():
            if from_state not in seen_states:
                all_states.append(from_state)
                seen_states.add(from_state)
            if to_state not in seen_states:
                all_states.append(to_state)
                seen_states.add(to_state)

        return all_states

    def get_possible_events(self, state_rm):
        """
        Return all events that can occur from a given Reward Machine state.

        :param state_rm: Reward Machine state.
        :return: List of events that have outgoing transitions from state_rm.
        """
        possible_events = []
        for (current_state, event), (new_state, _) in self.transitions.items():
            if current_state == state_rm:
                possible_events.append(event)
        return possible_events

    def get_current_state(self):
        """Return the current Reward Machine state."""
        return self.current_state

    def numbers_state(self):
        """
        Return the number of distinct Reward Machine states appearing in transitions.
        """
        states = set()
        for (from_state, _), (to_state, _) in self.transitions.items():
            states.add(from_state)
            states.add(to_state)
        return len(states)

    @property
    def get_transitions(self):
        """Property accessor for the transitions dictionary."""
        return self.transitions

    def reset_to_initial_state(self):
        """
        Reset the current state of the Reward Machine to its initial state.
        """
        self.current_state = self.initial_state
        return self.initial_state

    def get_final_state(self):
        """
        Return the last target state appearing in the transitions dictionary.

        If no transitions are defined, returns None.
        """
        if self.transitions:  # Ensure transitions exist
            last_to_state = next(reversed(self.transitions.values()))[0]
            return last_to_state
        else:
            # No transitions defined
            return None

    def _get_start_state(self):
        """
        Infer the start state as the source state of the first transition.

        If no transitions are defined, returns None.
        """
        if not self.transitions:
            return None

        # Take the source state of the first transition in the transitions dict
        first_transition = next(iter(self.transitions))
        start_state = first_transition[0]
        return start_state

    def get_state_from_index(self, rm_state_index):
        """
        Given the mapping self.state_indices = { rm_state: idx },
        return the rm_state (i.e., the key) corresponding to rm_state_index.

        :param rm_state_index: Integer index of the RM state.
        :return: The RM state label associated with rm_state_index.
        :raises ValueError: If the index is not present.
        """
        # Invert the dictionary on the fly. For efficiency, you could
        # store the inverse mapping as an attribute.
        inv_map = {v: k for k, v in self.state_indices.items()}
        if rm_state_index not in inv_map:
            raise ValueError(
                f"Index {rm_state_index} not present in RewardMachine.state_indices"
            )
        return inv_map[rm_state_index]

    def add_reward_shaping(self, gamma, rs_gamma):
        """
        Add potential-based reward shaping using value iteration over the RM.

        :param gamma: Discount factor for the environment.
        :param rs_gamma: Discount factor used in the shaping computation.
        """
        self.gamma = gamma
        self.potentials = self.value_iteration(
            list(self.state_indices.keys()),
            self.get_delta_u(),
            self.get_delta_r(),
            self.get_final_state(),
            rs_gamma,
        )
        # Negate potentials (common convention in some shaping setups)
        for u in self.potentials:
            self.potentials[u] = -self.potentials[u]

    def add_distance_reward_shaping(self, gamma, rs_gamma, alpha=100):
        """
        Add potential-based shaping where potentials are a (negative) function
        of the distance in the RM graph to the final state.

        :param gamma: Discount factor for the environment (unused here but kept for API symmetry).
        :param rs_gamma: Discount factor for shaping (unused here but kept for API symmetry).
        :param alpha: Scaling factor for distance-based potentials.
        """
        # Initialize self.potentials as an empty dict
        self.potentials = {}

        final_state = self.get_final_state()

        # Compute distance for every state
        all_states = self.get_all_states()
        for u in all_states:
            if u == final_state:
                # Potential is 0 for the final state
                self.potentials[u] = 0
            else:
                # get_distance(u) returns the minimum number of transitions
                dist = self.get_distance(u)
                # Define Phi(u) = - alpha * dist
                self.potentials[u] = -alpha * dist

    def get_distance(self, start_state):
        """
        Returns the minimum number of transitions needed to reach the final state
        from start_state, or a large number (e.g., 999999) if the final state is
        unreachable from start_state.

        Assumes there is exactly one final state, obtained via self.get_final_state().

        :param start_state: Starting RM state.
        :return: Minimum distance as an integer or 999999 if unreachable.
        """
        final_state = self.get_final_state()
        if start_state == final_state:
            return 0

        # BFS queue: each element is (current_state, distance)
        queue = deque([(start_state, 0)])
        visited = set([start_state])

        while queue:
            current, dist = queue.popleft()

            # Check if we reached the final state
            if current == final_state:
                return dist

            # Expand all successor states
            # self.transitions has the form: (u, event) -> (v, reward)
            # So we look for all transitions starting from current
            for (u, event), (v, rew) in self.transitions.items():
                if u == current and v not in visited:
                    visited.add(v)
                    # Distance of successor is dist+1
                    queue.append((v, dist + 1))

        # If the queue is exhausted without finding the final state, it's unreachable
        return 999999

    def get_delta_u(self):
        """
        Build the RM transition structure delta_u[u1][u2] = event,
        ensuring all states appear as keys even if they only occur as targets.
        """
        delta_u = {}
        for (u1, event), (u2, _) in self.transitions.items():
            if u1 not in delta_u:
                delta_u[u1] = {}
            if u2 not in delta_u:
                delta_u[u2] = {}  # Ensure u2 is also present
            delta_u[u1][u2] = event
        return delta_u

    def get_delta_r(self):
        """
        Build the RM reward structure delta_r[u1][u2] = ConstantRewardFunction(reward),
        ensuring all states appear as keys even if they only occur as targets.
        """
        delta_r = {}
        for (u1, event), (u2, reward) in self.transitions.items():
            if u1 not in delta_r:
                delta_r[u1] = {}
            if u2 not in delta_r:
                delta_r[u2] = {}  # Ensure u2 is also present
            delta_r[u1][u2] = ConstantRewardFunction(reward)
        return delta_r

    def value_iteration(self, U, delta_u, delta_r, terminal_u, gamma):
        """
        Classic value iteration over the RM state space.

        :param U: List of RM states.
        :param delta_u: Transition mapping: delta_u[u1][u2] = event.
        :param delta_r: Reward mapping: delta_r[u1][u2] = RewardFunction instance.
        :param terminal_u: Terminal RM state.
        :param gamma: Discount factor for value iteration.
        :return: Dictionary V[u] with the value of each RM state.
        """
        V = dict([(u, 0) for u in U])
        V[terminal_u] = 0
        V_error = 1

        # Debug print statements
        print("Initial V:", V)
        print("Delta U:", delta_u)
        print("Delta R:", delta_r)

        while V_error > 0.0000001:
            V_error = 0
            for u1 in U:
                if not delta_u[u1]:  # Skip states with no outgoing transitions
                    continue
                q_u2 = []
                for u2 in delta_u[u1]:
                    if delta_r[u1][u2].get_type() == "constant":
                        r = delta_r[u1][u2].get_reward(None)
                    else:
                        # If the reward function is not constant, assume zero
                        r = 0
                    q_u2.append(r + gamma * V[u2])
                if q_u2:  # Ensure q_u2 is not empty
                    v_new = max(q_u2)
                    V_error = max([V_error, abs(v_new - V[u1])])
                    V[u1] = v_new
        return V


# Definition of RewardFunction hierarchy
class RewardFunction:
    def __init__(self):
        pass

    def get_reward(self, s_info):
        raise NotImplementedError("To be implemented")

    def get_type(self):
        raise NotImplementedError("To be implemented")


class ConstantRewardFunction(RewardFunction):
    """
    Reward function that always returns a constant value.
    """

    def __init__(self, c):
        super().__init__()
        self.c = c

    def get_type(self):
        return "constant"

    def get_reward(self, s_info):
        return self.c
