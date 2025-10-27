import numpy as np
from typing import Optional
from multiagent_rlrm.learning_algorithms.learning_algorithm import BaseLearningAlgorithm


class UCBVI(BaseLearningAlgorithm):
    def __init__(
        self,
        state_space_size: int,
        action_space_size: int,
        ep_len: int = 100,
        gamma: float = 1.0,
        bonus_scaling: float = 1.0,
        bonus_type: str = "simplified_bernstein",
        reward_free: bool = False,
        stage_dependent: bool = False,
        real_time_dp: bool = False,
        debug_interval: int = 0,
        seed: int = 0,
        reward_range: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(state_space_size, action_space_size, seed)

        assert 0.0 < gamma <= 1.0
        if gamma < 1.0 and ep_len is None:
            ep_len = int(np.ceil(1.0 / (1.0 - gamma)))
        self.H = ep_len
        self.gamma = gamma

        # flags
        self.c = bonus_scaling
        self.bonus_type = bonus_type
        self.reward_free = reward_free
        self.stage_dependent = stage_dependent
        self.real_time_dp = real_time_dp
        self.debug_interval = debug_interval

        self.param_str = f"H{ep_len}_c{bonus_scaling}_{bonus_type}"

        S, A, H = state_space_size, action_space_size, self.H
        if stage_dependent:
            shape_hsa = (H, S, A)
            shape_hsas = (H, S, A, S)
        else:
            shape_hsa = (S, A)
            shape_hsas = (S, A, S)

        # counts & model
        self.N_sa = np.zeros(shape_hsa, dtype=np.int32)
        self.R_hat = np.zeros(shape_hsa, dtype=np.float32)
        self.P_hat = np.ones(shape_hsas, dtype=np.float32) / S

        # bonuses and value tables
        self.B_sa = np.zeros((H, S, A), dtype=np.float32)
        self.V = np.zeros((H, S), dtype=np.float32)
        self.Q = np.zeros((H, S, A), dtype=np.float32)

        self.V_policy = np.zeros((H, S), dtype=np.float32)
        self.Q_policy = np.zeros((H, S, A), dtype=np.float32)

        # reward span
        if reward_range is None:
            r_span = 1.0
        else:
            r_span = abs(reward_range[1] - reward_range[0]) or 1.0

        # v_max
        self.v_max = np.zeros(self.H, dtype=np.float32)
        self.v_max[-1] = r_span
        for h in range(self.H - 2, -1, -1):
            self.v_max[h] = r_span + gamma * self.v_max[h + 1]

        # optimistic init
        for h in range(H):
            self.V[h, :] = self.v_max[h]
            self.B_sa[h, :, :] = self.v_max[h]

        self.tstep = 0
        self.episode = 0

        self.tosave += [
            "N_sa",
            "R_hat",
            "P_hat",
            "B_sa",
            "Q",
            "V",
            "Q_policy",
            "V_policy",
        ]

    # ---------------- policy ----------------
    def choose_action(
        self,
        encoded_state: int,
        best: bool = False,
        rng: Optional[np.random.Generator] = None,
        **kwargs,
    ) -> int:
        h = min(self.tstep, self.H - 1)
        if best:
            return int(np.argmax(self.Q_policy[h, encoded_state]))

        q_row = self.Q[h, encoded_state]
        rng = self.rng if rng is None else rng
        maxv = q_row.max()
        idxs = np.flatnonzero(q_row == maxv)
        return int(rng.choice(idxs))

    # --------------- update ------------------
    def update(
        self,
        encoded_state: int,
        encoded_next_state: int,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool = False,
        **kwargs,
    ) -> bool:
        done = terminated or truncated
        h = min(self.tstep, self.H - 1)

        if self.stage_dependent:
            self.N_sa[h, encoded_state, action] += 1
            nn = self.N_sa[h, encoded_state, action]
            self._inc_update_model_sd(
                h, encoded_state, action, encoded_next_state, reward, nn
            )
        else:
            self.N_sa[encoded_state, action] += 1
            nn = self.N_sa[encoded_state, action]
            self._inc_update_model(
                encoded_state, action, encoded_next_state, reward, nn, h=h
            )

        self.tstep += 1
        timeout = self.tstep >= self.H
        episode_end = done or timeout

        if episode_end:
            if not self.real_time_dp:
                self._plan_full_backup()

            if self.debug_interval and self.episode % self.debug_interval == 0:
                print(
                    f"[UCBVI] episode {self.episode} ▸ V0={self.V[0,0]:.3f}  N_visits={self.N_sa.sum()}"
                )

            self.tstep = 0
            self.episode += 1

        return episode_end

    # ---------- internal helpers ----------
    def _inc_update_model(self, s, a, s_next, r, n, h=None):
        prev_r = self.R_hat[s, a]
        prev_p = self.P_hat[s, a]
        eta = 1.0 / n
        self.R_hat[s, a] = (1.0 - eta) * prev_r + eta * r
        self.P_hat[s, a] = (1.0 - eta) * prev_p
        self.P_hat[s, a, s_next] += eta

        if h is not None:
            if h + 1 < self.H:
                pVsq = np.inner(self.P_hat[s, a], self.V[h + 1] ** 2)
                pV = np.inner(self.P_hat[s, a], self.V[h + 1])
                var = np.clip(pVsq - pV ** 2, 0.0, self.v_max[h] ** 2)
            else:
                var = 0.0
            self.B_sa[h, s, a] = self._compute_bonus(n, h, var)

    def _inc_update_model_sd(self, h, s, a, s_next, r, n):
        prev_r = self.R_hat[h, s, a]
        prev_p = self.P_hat[h, s, a]
        eta = 1.0 / n
        self.R_hat[h, s, a] = (1.0 - eta) * prev_r + eta * r
        self.P_hat[h, s, a] = (1.0 - eta) * prev_p
        self.P_hat[h, s, a, s_next] += eta

        if h + 1 < self.H:
            pVsq = np.inner(self.P_hat[h, s, a], self.V[h + 1] ** 2)
            pV = np.inner(self.P_hat[h, s, a], self.V[h + 1])
            var = np.clip(pVsq - pV ** 2, 0.0, self.v_max[h] ** 2)
        else:
            var = 0.0
        self.B_sa[h, s, a] = self._compute_bonus(n, h, var)

    def _compute_bonus(self, n: int, h: int, var: float | None = None) -> float:
        if n <= 0:
            return self.v_max[h]
        if self.reward_free:
            return 1.0 / n

        if self.bonus_type == "simplified_bernstein":
            bonus = self.c * np.sqrt(1.0 / n) + self.v_max[h] / n
        elif self.bonus_type == "bernstein":
            var = 0.0 if var is None else var
            bonus = self.c * (np.sqrt(var / n) + self.v_max[h] / n)
        elif self.bonus_type == "hoeffding":
            bonus = self.c * (np.sqrt(1.0 / n) * (self.v_max[h] / 2.0))
        else:
            raise ValueError(f"bonus_type {self.bonus_type} non implementato")

        return min(bonus, self.v_max[h])

    def _plan_full_backup(self) -> None:
        H, S, A = self.H, self.state_space_size, self.action_space_size
        V_next = np.zeros(S, dtype=np.float32)
        for h in range(H - 1, -1, -1):
            if self.stage_dependent:
                R = self.R_hat[h]
                P = self.P_hat[h]
            else:
                R = self.R_hat
                P = self.P_hat
            bonus = self.B_sa[h]
            Q_h = R + bonus + self.gamma * (P @ V_next)
            Q_h = np.minimum(Q_h, self.v_max[h])
            self.Q[h] = Q_h
            self.V[h] = Q_h.max(axis=1)
            V_next = self.V[h]
        self.compute_greedy_policy()
        self.q_table = self.Q_policy[0].copy()

    def compute_greedy_policy(self) -> None:
        H, S, A = self.H, self.state_space_size, self.action_space_size
        V_next = np.zeros(S, dtype=np.float32)
        for h in range(H - 1, -1, -1):
            if self.stage_dependent:
                R = self.R_hat[h]
                P = self.P_hat[h]
            else:
                R = self.R_hat
                P = self.P_hat
            Q_h = R + self.gamma * (P @ V_next)
            Q_h = np.minimum(Q_h, self.v_max[h])
            self.Q_policy[h] = Q_h
            self.V_policy[h] = Q_h.max(axis=1)
            V_next = self.V_policy[h]

    def greedy_action(self, state: int, hh: int = 0) -> int:
        return int(np.argmax(self.Q_policy[hh, state]))
