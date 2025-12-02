import numpy as np
from typing import Optional
from multiagent_rlrm.learning_algorithms.learning_algorithm import BaseLearningAlgorithm


# ============================================================
#  Backward induction utilities (pure numpy, no numba)
# ============================================================


def backward_induction_in_place(Q, V, R, P, horizon, gamma=1.0, vmax=np.inf):
    """
    R: (S,A,B)    P: (S,A,S,B)   Q: (H,S,A)   V: (H,S)
    """
    S, A, B = R.shape
    V_next = np.zeros(S, dtype=np.float32)
    for hh in range(horizon - 1, -1, -1):
        # exp_val[s,a,b] = sum_ns P[s,a,ns,b] * V_next[ns]
        exp_val = np.tensordot(P, V_next, axes=([2], [0]))  # -> (S,A,B)
        Q_b = R + gamma * exp_val  # (S,A,B)
        Q_h = Q_b.max(axis=2)  # max over B
        Q[hh] = Q_h
        V_h = Q_h.max(axis=1)
        if vmax != np.inf:
            np.minimum(V_h, vmax, out=V_h)
        V[hh] = V_h
        V_next = V_h


def backward_induction_sd(Q, V, R, P, gamma=1.0, vmax=np.inf):
    """
    R: (H,S,A,B)  P: (H,S,A,S,B)
    """
    H, S, A, B = R.shape
    V_next = np.zeros(S, dtype=np.float32)
    for hh in range(H - 1, -1, -1):
        exp_val = np.tensordot(P[hh], V_next, axes=([3], [0]))  # (S,A,B)
        Q_b = R[hh] + gamma * exp_val
        Q_h = Q_b.max(axis=2)
        Q[hh] = Q_h
        V_h = Q_h.max(axis=1)
        if vmax != np.inf:
            np.minimum(V_h, vmax, out=V_h)
        V[hh] = V_h
        V_next = V_h


# ============================================================
#  OPSRL (Posterior Sampling with Beta/Dirichlet priors + optimistic dummy state)
# ============================================================


class OPSRL(BaseLearningAlgorithm):
    """
    Optimistic Posterior Sampling for RL with:
      - Bernoulli-ized rewards and Beta(scale*(1,1)) priors.
      - Dirichlet transitions with 'uniform' or 'optimistic' (dummy S+1) priors.
      - Thompson sampling with B samples per episode.
      - Optional absorbing terminal states (RMAX/QRMAX style).
      - Same choose_action/update interface as other algorithms (returns bool at episode end).

    Parameters
    ----------
    state_space_size : int
    action_space_size : int
    ep_len : int = 100
    gamma : float = 1.0
    bernoullized_reward : bool = True
    scale_prior_reward : float = 1.0
    thompson_samples : int = 1
    prior_transition : {'uniform', 'optimistic'} = 'uniform'
    scale_prior_transition : float | None = None
    reward_free : bool = False
    stage_dependent : bool = False
    make_absorbing_on_done : bool = False
    absorb_alpha : float = 1e6
    debug_interval : int = 0
    seed : int = 0
    """

    # ----------------------------------------------------------- init --------
    def __init__(
        self,
        state_space_size: int,
        action_space_size: int,
        ep_len: int = 100,
        gamma: float = 1.0,
        bernoullized_reward: bool = True,
        scale_prior_reward: float = 1.0,
        thompson_samples: int = 1,
        prior_transition: str = "uniform",
        scale_prior_transition: Optional[float] = None,
        reward_free: bool = False,
        stage_dependent: bool = False,
        make_absorbing_on_done: bool = False,
        absorb_alpha: float = 1e6,
        debug_interval: int = 0,
        seed: int = 0,
    ) -> None:

        super().__init__(state_space_size, action_space_size, seed)

        assert 0.0 < gamma <= 1.0
        assert prior_transition in ("uniform", "optimistic")

        self.H = ep_len
        self.gamma = gamma
        self.bernoullized_reward = bernoullized_reward
        self.scale_prior_reward = scale_prior_reward
        self.thompson_samples = thompson_samples
        self.prior_transition = prior_transition
        self.scale_prior_transition = scale_prior_transition
        self.reward_free = reward_free
        self.stage_dependent = stage_dependent
        self.make_absorbing_on_done = make_absorbing_on_done
        self.absorb_alpha = absorb_alpha
        self.debug_interval = debug_interval

        S, A, H = state_space_size, action_space_size, ep_len

        # ---------------- v_max (upper bound) ----------------
        # If reward range is unknown, assume [0,1]
        r_span = 1.0
        self.v_max = np.zeros(H, dtype=np.float32)
        self.v_max[-1] = r_span
        for h in range(H - 2, -1, -1):
            self.v_max[h] = r_span + gamma * self.v_max[h + 1]

        # ---------------- shapes ----------------
        if stage_dependent:
            shp_hsa = (H, S, A)
            shp_hsas = (H, S, A, S)
        else:
            shp_hsa = (S, A)
            shp_hsas = (S, A, S)

        # dummy state se optimistic
        self.use_dummy_state = prior_transition == "optimistic"
        if self.use_dummy_state:
            S_next = S + 1
            if stage_dependent:
                shp_hsas = (H, S, A, S_next)
            else:
                shp_hsas = (S, A, S_next)
        else:
            S_next = S

        self.S = S
        self.A = A
        self.S_next = S_next

        # ---------------- Priors ----------------
        if self.scale_prior_transition is None:
            self.scale_prior_transition = (
                1.0 / S if prior_transition == "uniform" else 1.0
            )

        # Dirichlet counts
        if prior_transition == "uniform":
            self.N_sas = np.full(
                shp_hsas, self.scale_prior_transition, dtype=np.float64
            )
        else:
            self.N_sas = np.zeros(shp_hsas, dtype=np.float64)
            # tutta la massa iniziale sul dummy
            self.N_sas[..., -1] = self.scale_prior_transition

        # Beta counts (a,b)
        self.M_sa = np.full(shp_hsa + (2,), self.scale_prior_reward, dtype=np.float64)

        # ---------------- Value / Q ----------------
        # V tiene S_next (dummy incluso); Q solo stati reali
        self.V = np.zeros((H, S_next), dtype=np.float32)
        self.Q = np.zeros((H, S, A), dtype=np.float32)

        # optimistic init: evita 1/(1-gamma) quando gamma=1
        if self.use_dummy_state:
            for h in range(H):
                self.V[h, :] = self.v_max[h]

        # policy greedy (MAP)
        self.V_policy = np.zeros((H, S), dtype=np.float32)
        self.Q_policy = np.zeros((H, S, A), dtype=np.float32)

        # absorbing mask solo sugli S reali
        self.absorbing = np.zeros(S, dtype=bool)

        # bookkeeping
        self.tstep = 0
        self.episode = 0
        self._planned_for_episode = -1  # flag for TS planning

        # cosa salvare
        self.tosave += [
            "N_sas",
            "M_sa",
            "V",
            "Q",
            "V_policy",
            "Q_policy",
            "absorbing",
        ]
        self.param_str = f"H{H}_B{thompson_samples}_{prior_transition}"

    # ------------------------------------------------- POLICY ---------------
    def choose_action(
        self,
        encoded_state: int,
        best: bool = False,
        rng: Optional[np.random.Generator] = None,
        **kwargs,
    ) -> int:
        """
        Select an action using the Thompson-sampled policy for this episode.

        • best=False → training policy (uses Q from current episode TS)
        • best=True  → greedy policy frozen for evaluation (MAP)
        """
        h = min(self.tstep, self.H - 1)
        if rng is None:
            rng = self.rng

        if best:
            row = self.Q_policy[h, encoded_state]
        else:
            # If we have not yet sampled/planned for this episode
            if self._planned_for_episode != self.episode:
                self._sample_and_plan()
            row = self.Q[h, encoded_state]

        maxv = row.max()
        idxs = np.flatnonzero(row == maxv)
        return int(rng.choice(idxs))

    # ------------------------------------------------- UPDATE ---------------
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
        """
        Update Beta/Dirichlet posteriors and re-plan when an episode ends.

        If terminated and make_absorbing_on_done -> make next_state absorbing.
        Returns True when the episode is finished.
        """
        done = terminated or truncated
        h = min(self.tstep, self.H - 1)

        # Bernoulli-ize reward (or zero it if reward_free)
        r_obs = reward
        if self.reward_free:
            r_obs = 0.0
        elif self.bernoullized_reward:
            p = np.clip(reward, 0.0, 1.0)
            r_obs = self.rng.binomial(1, p)

        # ---- Beta update ----
        if self.stage_dependent:
            self.M_sa[h, encoded_state, action, 0] += r_obs
            self.M_sa[h, encoded_state, action, 1] += 1 - r_obs
        else:
            self.M_sa[encoded_state, action, 0] += r_obs
            self.M_sa[encoded_state, action, 1] += 1 - r_obs

        # ---- Dirichlet update ----
        ns = encoded_next_state
        if self.stage_dependent:
            self.N_sas[h, encoded_state, action, ns] += 1.0
        else:
            self.N_sas[encoded_state, action, ns] += 1.0

        # Absorbing states (holes)
        if done and self.make_absorbing_on_done:
            self._make_absorbing(ns)

        # step/episode counters
        self.tstep += 1
        timeout = self.tstep >= self.H
        episode_end = done or timeout

        if episode_end:
            # aggiorna policy MAP (per i test)
            self._plan_map_policy()

            if self.debug_interval and (self.episode % self.debug_interval == 0):
                print(f"[OPSRL] episode {self.episode} ▸ V0={self.V_policy[0,0]:.3f}")

            self.tstep = 0
            self.episode += 1

        return episode_end

    # ------------------------------------------------- INTERNALS ------------

    def _make_absorbing(self, s_abs: int):
        """Make s_abs absorbing (only if it is a real state < S)."""
        if s_abs >= self.S or self.absorbing[s_abs]:
            return
        self.absorbing[s_abs] = True

        # reward -> 0 con altissima "certezza"
        if self.stage_dependent:
            self.M_sa[:, s_abs, :, 0] = 1.0
            self.M_sa[:, s_abs, :, 1] = 1e12
        else:
            self.M_sa[s_abs, :, 0] = 1.0
            self.M_sa[s_abs, :, 1] = 1e12

        # transizioni -> self-loop massiccio
        if self.stage_dependent:
            self.N_sas[:, s_abs, :, :] = 0.0
            self.N_sas[:, s_abs, :, s_abs] = self.absorb_alpha
        else:
            self.N_sas[s_abs, :, :] = 0.0
            self.N_sas[s_abs, :, s_abs] = self.absorb_alpha

    def _sample_and_plan(self):
        """Thompson sampling: draw B MDPs and plan via backward induction."""
        B = self.thompson_samples
        H, S, A, S_next = self.H, self.S, self.A, self.S_next

        # -------- sample rewards --------
        if self.stage_dependent:
            a = np.repeat(self.M_sa[..., 0][..., None], B, -1)  # (H,S,A,B)
            b = np.repeat(self.M_sa[..., 1][..., None], B, -1)
            R_samp = self.rng.beta(a, b)
        else:
            a = np.repeat(self.M_sa[..., 0][..., None], B, -1)  # (S,A,B)
            b = np.repeat(self.M_sa[..., 1][..., None], B, -1)
            R_tmp = self.rng.beta(a, b)  # (S,A,B)
            R_samp = np.repeat(R_tmp[None, ...], self.H, axis=0)  # (H,S,A,B) scrivibile

        # -------- sample transitions --------
        if self.stage_dependent:
            alpha = np.repeat(self.N_sas[..., None], B, -1)  # (H,S,A,S_next,B)
            P_samp = self.rng.gamma(alpha)
            P_samp /= P_samp.sum(axis=3, keepdims=True)
            P_for_bi = P_samp[..., : self.S, :] if self.use_dummy_state else P_samp
        else:
            alpha = np.repeat(self.N_sas[..., None], B, -1)  # (S,A,S_next,B)
            P_tmp = self.rng.gamma(alpha)
            P_tmp /= P_tmp.sum(axis=2, keepdims=True)
            P_samp = np.repeat(P_tmp[None, ...], self.H, axis=0)  # (H,S,A,S_next,B)
            P_for_bi = (
                P_samp[0, ..., : self.S, :] if self.use_dummy_state else P_samp[0]
            )

        # -------- enforce absorbing states --------
        if self.absorbing.any():
            abs_idx = np.where(self.absorbing)[0]
            # rewards = 0
            R_samp[:, abs_idx, :, :] = 0.0
            # transizioni self-loop
            if self.stage_dependent:
                P_for_bi[:, abs_idx, :, :, :] = 0.0
                for s in abs_idx:
                    P_for_bi[:, s, :, s, :] = 1.0
            else:
                P_for_bi[abs_idx, :, :, :] = 0.0
                for s in abs_idx:
                    P_for_bi[s, :, s, :] = 1.0

        # -------- backward induction --------
        if self.stage_dependent:
            backward_induction_sd(
                self.Q,
                self.V[:, : self.S],  # ignoriamo dummy nel backup
                R_samp,
                P_for_bi,
                self.gamma,
                np.inf,
            )
        else:
            backward_induction_in_place(
                self.Q,
                self.V[:, : self.S],
                R_samp[0],  # (S,A,B)
                P_for_bi,  # (S,A,S,B)
                H,
                self.gamma,
                np.inf,
            )

        self._planned_for_episode = self.episode

    def _plan_map_policy(self):
        """Policy greedy su modello MAP (mean posterior), senza Thompson."""
        H, S, A, S_next = self.H, self.S, self.A, self.S_next

        # ---- MAP rewards ----
        R_hat_base = self.M_sa[..., 0] / (
            self.M_sa[..., 0] + self.M_sa[..., 1]
        )  # (S,A) o (H,S,A)
        if self.stage_dependent:
            R_hat = R_hat_base
        else:
            # crea array (H,S,A) scrivibile
            R_hat = np.repeat(R_hat_base[None, ...], self.H, axis=0).astype(np.float64)

        # ---- MAP transitions ----
        sums = self.N_sas.sum(axis=-1, keepdims=True)
        P_hat_base = self.N_sas / np.clip(
            sums, 1e-12, None
        )  # (S,A,S_next) o (H,S,A,S_next)
        if self.stage_dependent:
            P_hat = P_hat_base
        else:
            P_hat = np.repeat(P_hat_base[None, ...], self.H, axis=0).astype(np.float64)

        if self.absorbing.any():
            abs_idx = np.where(self.absorbing)[0]
            R_hat[:, abs_idx, :] = 0.0
            P_hat[:, abs_idx, :, :] = 0.0
            for s in abs_idx:
                P_hat[:, s, :, s] = 1.0

        # backward induction semplice (niente bootstrap, max solo sulle azioni)
        V_next = np.zeros(S, dtype=np.float32)
        for h in range(H - 1, -1, -1):
            # ignora dummy se esiste
            P_slice = P_hat[h, :, :, :S]
            Q_h = R_hat[h] + self.gamma * (P_slice @ V_next)
            self.Q_policy[h] = Q_h
            self.V_policy[h] = Q_h.max(axis=1)
            V_next = self.V_policy[h]

        # compatibilità con codice esterno
        self.q_table = self.Q_policy[0].copy()

    # ------------------------------------------------- helper ---------------
    def greedy_action(self, state: int, hh: int = 0) -> int:
        """Azione della policy raccomandata (MAP)."""
        return int(np.argmax(self.Q_policy[hh, state]))
