import time, math, sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import gymnasium

# import frozen_lake
# from env.frozen_lake import *

from multiagent_rlrm.learning_algorithms.learning_algorithm import BaseLearningAlgorithm


# Q-Rmax
class QRMax_v2(BaseLearningAlgorithm):
    """Model-based QRMax algorithm supporting Reward Machine state augmentation."""

    def __init__(
        self,
        max_reward: float = 1.0,
        q_space_size=1,
        nsamplesTE=100,
        nsamplesRE=100,
        nsamplesTQ=1,
        nsamplesRQ=1,
        VI_delta=1e-4,
        VI_delta_rel=False,  # wheather delta in VI is relative to current values
        use_qrm=False,  # Use qrm experience
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.nA = self.action_space_size  # A size
        self.nSQ = self.state_space_size  # SxQ size
        self.nQ = q_space_size  # Q size
        self.nS = int(self.nSQ / self.nQ)  # S size

        self.R_max = max_reward * self.nQ

        self.use_qrm = use_qrm

        # value iteration parameters
        self.delta_value_iter = VI_delta
        self.delta_rel = VI_delta_rel
        self.max_num_value_iter = 1e4

        self.reset_VI = False

        self.Q = np.ones((self.nS, self.nQ, self.nA)) * self.R_max / (1 - self.gamma)

        # P(s',q'|s,q,a)
        self.nTSQA = np.zeros((self.nS, self.nQ, self.nA))  # n. of observed transitions
        # self.nTSQASQ = np.zeros((self.nS, self.nQ, self.nA, self.nS, self.nQ))  # observed transitions to new state
        self.nTSQA_dict = {}  # set of new states from (s,q,a)

        # E[r(s,q,a)]
        self.nRSA = np.zeros((self.nS, self.nA))  # n. of observed rewards
        self.nRQS = np.zeros((self.nQ, self.nS))  # n. of observed rewards
        self.nRSQA = np.zeros((self.nS, self.nQ, self.nA))  # n. of observed rewards
        self.nRSQASQ = np.zeros(
            (self.nS, self.nQ, self.nA, self.nS, self.nQ)
        )  # n. of observed rewards to new state
        self.sumR = np.zeros(
            (self.nS, self.nQ, self.nA, self.nS, self.nQ)
        )  # sum of observed values of reward
        self.sumRSA = np.zeros((self.nS, self.nA))  # sum of observed values of reward
        self.sumRQS = np.zeros((self.nQ, self.nS))  # sum of observed values of reward

        # P(q'|s',q)
        self.nTQSQ = np.zeros((self.nQ, self.nS, self.nQ))  # (q,s',q')
        self.nTQS = np.zeros((self.nQ, self.nS))  # (q,s')
        self.nTQSQ_dict = {}  # set of (q,s') -> q'
        self.entrySQ_dict = {}  # nr. of entry state (s',q') visits

        # P(s'|s,a)
        self.nTSAS = np.zeros((self.nS, self.nA, self.nS))  # (s,a,s')
        self.nTSA = np.zeros((self.nS, self.nA))  # (s,a)
        self.nTSAS_dict = {}  # set of (s,a) -> s'
        self.pTSAS = np.zeros((self.nS, self.nA, self.nS))  # p(s'|s,a)

        # P(q'|s',s,q,a)
        # P(s'|s,q,a)
        self.nTSQAS = np.zeros((self.nS, self.nQ, self.nA, self.nS))  # (s,q,a,s')

        self.knownTSQA = np.zeros((self.nS, self.nQ, self.nA))
        self.knownRSQA = np.zeros((self.nS, self.nQ, self.nA))
        self.knownTSA = np.zeros((self.nS, self.nA))
        self.knownTQS = np.zeros((self.nQ, self.nS))  # (q, s') known
        self.knownRSA = np.zeros((self.nS, self.nA))
        self.knownRQS = np.zeros((self.nQ, self.nS))

        self.final_states = []  # observed final states

        self.nsamplesTE = nsamplesTE
        self.nsamplesRE = nsamplesRE
        self.nsamplesTQ = nsamplesTQ
        self.nsamplesRQ = nsamplesRQ

        self.VI_iterations = 0
        self.noupdate_cnt = 0
        self.goal_found = False
        self.quit_experiment = -1  # quit after n episodes (if not -1)

        self.verbose = 0

        self.param_str = f"{self.nsamplesTE},{self.gamma:.2f},{self.delta_value_iter},{self.delta_rel}"

        print(f"!!! QRMax v2 ({self.param_str}) !!!")

        self.tosave += [
            "Q",
            "nTSQA",
            "nTSQA_dict",
            "nRSA",
            "nRQS",
            "nRSQA",
            "nRSQASQ",
            "nTQSQ",
            "nTQS",
            "nTQSQ_dict",
            "entrySQ_dict",
            "nTSAS",
            "nTSA",
            "nTSAS_dict",
            "pTSAS",
            "nTSQAS",
            "sumR",
            "sumRSA",
            "sumRQS",
            "knownTSQA",
            "knownRSQA",
            "knownTSA",
            "knownTQS",
            "knownRSA",
            "knownRQS",
            "final_states",
            "VI_iterations",
        ]

        """
        # R-max paper
        N = self.nS  # number of states
        k = self.nA  # number of actions
        eps = 0.1  # error bound (near-optimality)
        delta = 0.1 # failure probability
        Rmax = self.R_max # R max value
        T = Rmax*math.pow(self.gamma,2*N)  # ???  # eps-return mixing time of an optimal policy.

        K1 = max( math.ceil(math.pow(4 * N * T * Rmax / eps, 3)),  -6 * math.pow( math.log( delta/(6*N*math.pow(k,2))),3) ) + 1

        print("K1 = %.3f" %K1)
        """

    """def choose_action(self, obs, best=True, rng=None, **kwargs) -> int:
        # print(kwargs, "Ooo")
        if self.nQ == 1:

            s, q = obs, 0
        else:
            s, q = kwargs["info"]["s"], kwargs["info"]["q"]

        # first best action
        action = np.argmax(self.Q[s, q])


        if rng is None:
            rng = self.rng
        # choose random action among best ones
        Qa = self.Q[s,q]
        va = np.argmax(Qa)
        maxs = [i for i,v in enumerate(Qa) if v == Qa[va]]
        if self.verbose>2:
            print(" obs ", obs)
            print(" ... Qa = ",Qa,"  va = ",va,"  maxs = ",maxs)
        action = rng.choice(maxs)


        return action"""

    def save_environment(self, filepath):
        """
        Persist the learned environment dynamics to disk.
        """
        environment_state = {
            "nTSAS": self.nTSAS,  # Count of observed transitions (s, a, s').
            "nTSA": self.nTSA,  # Count of observed transitions (s, a).
            "nTSAS_dict": self.nTSAS_dict,  # Map (s, a) to observed next states.
            "pTSAS": self.pTSAS,  # Estimated transition probabilities p(s'|s,a).
            "knownTSA": self.knownTSA,  # Flag if a transition (s, a) is known.
            # "sumRSA": self.sumRSA,          # Sum of observed rewards for each (s, a).
            # "nRSA": self.nRSA,              # Number of observed rewards r(s, a).
            # "knownRSA": self.knownRSA,      # Flag if reward r(s, a) is known.
        }
        with open(filepath, "wb") as f:
            pickle.dump(environment_state, f)
        print(f"Environment dynamics saved to {filepath}")

    def load_environment(self, filepath):
        """
        Load environment dynamics from disk.
        """
        with open(filepath, "rb") as f:
            environment_state = pickle.load(f)

        self.nTSAS = environment_state["nTSAS"]
        self.nTSA = environment_state["nTSA"]
        self.nTSAS_dict = environment_state["nTSAS_dict"]
        self.pTSAS = environment_state["pTSAS"]
        self.knownTSA = environment_state["knownTSA"]
        # self.sumRSA = environment_state["sumRSA"]
        # self.nRSA = environment_state["nRSA"]
        # self.knownRSA = environment_state["knownRSA"]

        print(f"Dinamiche dell'ambiente caricate da {filepath}")

    def choose_action(self, obs, best=False, rng=None, **kwargs):
        s, q = kwargs["info"]["s"], kwargs["info"]["q"]
        if rng is None:
            rng = self.rng

        actions = list(range(self.nA))  # List of all possible actions
        rng.shuffle(actions)
        Qa = self.Q[s, q]  # Q-values for the current state and 'q'

        if best == False and np.all(Qa == Qa[0]):
            # If not looking for the best action and all Q-values are equal, choose a random action
            action = rng.choice(actions)
        else:
            # Otherwise, choose the action with the highest Q-value
            action = np.argmax(Qa)

        return action

    # solve current MDP with value-iteration

    """
    V(s,q) = max_a { E[r(s,q,a)] + gamma * sum_{s',q'} P(s',q'|s,q,a) V(s',q') }
    Q(s,q,a) = E[r(s,q,a)] + gamma sum_{s',p'} P(s',q'|s,q,a) V(s',q')
    V(s,q) = max_a Q(s,q,a)

    E[r(s,q,a)] = sum_{s',q'} P(s',q'|s,q,a) r(s,q,a,s',q')
    P(s',q'|s,q,a) = P(q'|s',s,q,a) P(s'|s,q,a) = P(q'|s',q) P(s'|s,a)


    E[r(s,q,a)] = sum_{s',q'} P(q'|s',q) P(s'|s,a) r(s,q,a,s',q')
      = sum_{s'} P(s'|s,a)  sum_{q'} P(s'|s,a) [ r(s,a,s') + r(q,s',q) ]

    """

    def known(self, s, q, a):
        r1 = (
            self.knownTSA[s, a] == 1
            and self.knownRSA[s, a] == 1
            and self.knownTSQA[s, q, a] == 1
        )

        if self.verbose > 2:
            print(
                "known: %d %d %d"
                % (self.knownTSA[s, a], self.knownRSA[s, a], self.knownTSQA[s, q, a])
            )

        return r1

    def solve_VI(self):

        if self.verbose > 1:
            print("*", end="")
            sys.stdout.flush()

        self.VI_solved += 1

        if self.reset_VI:
            self.Q = (
                np.ones((self.nS, self.nQ, self.nA)) * self.R_max / (1 - self.gamma)
            )

        # Imposta i valori per gli stati finali
        for s, q in self.final_states:
            self.Q[s, q, :] = 0  # Vettorializzazione

        Qnew = np.copy(self.Q)
        i = 0
        delta = 1.0

        while delta > self.delta_value_iter and i < self.max_num_value_iter:
            delta = 0
            for s in range(self.nS):
                for q in range(self.nQ):
                    for a in range(self.nA):
                        if self.known(s, q, a) and (s, q) not in self.final_states:
                            qq1 = 0
                            qq2 = 0

                            # Pre-calcolo di valori comuni
                            rsa_div_nrsa = (
                                self.sumRSA[s, a] / self.nRSA[s, a]
                                if self.nRSA[s, a] > 0
                                else 0
                            )

                            for s1 in self.nTSAS_dict.get((s, a), []):
                                # Calcola una volta e riutilizza
                                Pssa = (
                                    self.nTSAS[s, a, s1] / self.nTSA[s, a]
                                    if self.nTSA[s, a] > 0
                                    else 0
                                )

                                for q1 in self.nTQSQ_dict.get((q, s1), set()):
                                    Pqsq = (
                                        self.nTQSQ[q, s1, q1] / self.nTQS[q, s1]
                                        if self.nTQS[q, s1] > 0
                                        else 0
                                    )
                                    tr = Pqsq * Pssa

                                    rqs_div_nrqs = (
                                        self.sumRQS[q, s1] / self.nRQS[q, s1]
                                        if self.nRQS[q, s1] > 0
                                        else 0
                                    )

                                    qq1 += tr * (rsa_div_nrsa + rqs_div_nrqs)
                                    qq2 += tr * np.max(self.Q[s1, q1])

                            Qnew[s, q, a] = qq1 + self.gamma * qq2
                            # use relative error for env with small Q values
                            cdelta = abs(Qnew[s, q, a] - self.Q[s, q, a])
                            if self.delta_rel:
                                m = abs(max(Qnew[s, q, a], self.Q[s, q, a]))
                                cdelta = cdelta / (m if m > 0 else 1)
                            delta = max(delta, cdelta)

            i += 1
            self.Q = np.copy(Qnew)

        self.VI_iterations += i

    def learn_init(self):
        self.VI_solved = 0
        self.goal_reached = 0

    def learn_init_episode(self):
        self.episode_steps = 0
        self.tosolve_VI = False

    def learn_done_episode(self):

        if self.tosolve_VI:
            self.solve_VI()

        nkTSA = np.sum(self.knownTSA)
        nkTQS = np.sum(self.knownTQS)
        nkRSA = np.sum(self.knownRSA)
        nkRQS = np.sum(self.knownRQS)

        """if self.episode % self.rleval.eval_ep_interval == 0:
            if self.verbose > 0:
                print(
                    "%05d %0.3f+/-%0.3f (known:%d,%d,%d,%d/%d - goal reached: %d %s - VI solved: %d/%d)"
                    % (
                        self.episode,
                        self.rleval.last_mean_r,
                        self.rleval.last_std_r,
                        nkTSA,
                        nkRSA,
                        nkTQS,
                        nkRQS,
                        self.nS * self.nQ * self.nA,
                        self.goal_reached,
                        "*"
                        if self.quit_experiment > 0
                        else "X"
                        if self.quit_experiment == 0
                        else " ",
                        self.VI_solved,
                        self.VI_iterations,
                    )
                )

                self.VI_solved = 0
                self.goal_reached = 0

                if self.verbose > 1:

                    for sq in self.entrySQ_dict.keys():
                        s1 = sq[0]
                        q1 = sq[1]

                        print(
                            " entry s1:%d q1:%d  %d - "
                            % (s1, q1, self.entrySQ_dict[(s1, q1)]),
                            end="",
                        )
                    print()"""

    def learn_end(self):
        pass

    def addTSQA(self, s, q, a, s1, q1):
        if (s, q, a) not in self.nTSQA_dict.keys():
            self.nTSQA_dict[(s, q, a)] = []
        if (s1, q1) not in self.nTSQA_dict[(s, q, a)]:
            self.nTSQA_dict[(s, q, a)].append((s1, q1))
            # if s==0 and a==3:
            #    print("\n  +++++ nTSQA_dict[%d,%d,%d]  append %d %d +++++\n" %(s,q,a,s1,q1))

        if self.knownTSQA[s, q, a] == 0:
            self.nTSQA[s, q, a] += 1
            self.nTSQAS[s, q, a, s1] += 1
            # self.nTSQASQ[s,q,a,s1,q1] += 1

    def update(self, obs, next_obs, action, reward, terminated, **kwargs):

        info = kwargs["info"]

        s, q = info["prev_s"], info["prev_q"]
        s1, q1 = info["s"], info["q"]
        a = action

        def update_RSQA_RSA(s, q, a, s1, q1, re, done):

            if done:
                if (s1, q1) not in self.final_states:
                    self.final_states.append((s1, q1))

            # entry states
            if q != q1:
                if (s1, q1) not in self.entrySQ_dict.keys():
                    self.entrySQ_dict[(s1, q1)] = 0
                self.entrySQ_dict[(s1, q1)] += 1

            # P(s',q'|s,q,a) = P(q'|s',q) P(s'|s,a)
            # self.verify_P(s,q,a,s1,q1)

            self.addTSQA(s, q, a, s1, q1)

            if self.knownTSA[s, a] == 0:
                self.nTSAS[s, a, s1] += 1
                self.nTSA[s, a] += 1

                if (s, a) not in self.nTSAS_dict.keys():
                    self.nTSAS_dict[(s, a)] = []
                if s1 not in self.nTSAS_dict[(s, a)]:
                    self.nTSAS_dict[(s, a)].append(s1)

            if self.knownRSA[s, a] == 0:
                self.nRSA[s, a] += 1
                self.sumRSA[s, a] += re

        def update_TQS_RQS(q, s1, q1, rq):

            if self.knownTQS[q, s1] == 0:
                self.nTQSQ[q, s1, q1] += 1
                self.nTQS[q, s1] += 1

                if (q, s1) not in self.nTQSQ_dict.keys():
                    self.nTQSQ_dict[(q, s1)] = []
                if q1 not in self.nTQSQ_dict[(q, s1)]:
                    self.nTQSQ_dict[(q, s1)].append(q1)

                assert len(self.nTQSQ_dict[(q, s1)]) == 1

            if self.knownRQS[q, s1] == 0:
                self.nRQS[q, s1] += 1
                self.sumRQS[q, s1] += rq

        re = info["Renv"]
        rq = info["RQ"]
        update_RSQA_RSA(s, q, a, s1, q1, re, terminated)
        update_TQS_RQS(q, s1, q1, rq)

        # check if some new info becomes known
        kk = self.check_known(s, q, a, s1, q1)

        if self.use_qrm:
            qrm_experiences = kwargs.get("info", {}).get("qrm_experience", [])
            for qrm_exp in qrm_experiences:
                _, _, _rtot, _, done, _s, _q, _s1, _q1, _rq = qrm_exp
                assert (
                    _rtot - _rq == re
                ), f"different rewards {re} != {_rtot} - {_rq} !!!"
                update_RSQA_RSA(_s, _q, a, _s1, _q1, re, done)
                update_TQS_RQS(_q, _s1, _q1, _rq)
                k1 = self.check_known(_s, _q, a, _s1, _q1)
                kk = kk or k1

        terminate = False
        self.noupdate_cnt += 1
        if self.noupdate_cnt % 10000 == 0:
            print(f"no update count: {self.noupdate_cnt}")
            if self.noupdate_cnt >= 1e6:
                terminate = True

        if kk and not self.user_quit:
            self.solve_VI()
            self.tosolve_VI = False  # will solve it at the end of the episode
            self.noupdate_cnt = 0

            # print(f"Known: {np.sum(self.knownTSA)}/{self.nS*self.nA} - {np.sum(self.knownTQS)}/{self.nQ*self.nS} ")

            # print(self.Q)
            # print("nTSA[0,1] = %d " %self.nTSA[0,1])

        return terminate  # not truncated

    def check_known(self, s, q, a, s1, q1):

        new_known = False

        if self.verbose > 2:
            print(
                "known: %d %d %d"
                % (self.knownTSA[s, a], self.knownRSA[s, a], self.knownTSQA[s, q, a])
            )

        # check transitions

        if self.knownTSQA[s, q, a] == 0:

            rT1 = self.nTSQA[s, q, a] >= self.nsamplesTE

            rT2 = False
            if self.knownTSA[s, a]:
                rT2 = True
                if self.verbose > 2 and (s == 12 and q == 1):
                    print("   --- check knownTSQA %d %d %d ---" % (s, q, a))
                for ss1 in self.nTSAS_dict[(s, a)]:
                    if self.knownTQS[q, ss1] == 1:
                        if self.verbose > 2 and (s == 12 and q == 1):
                            print("       +++  KnownTQS[%d,%d] --- " % (q, ss1), end="")
                            print(self.nTQSQ_dict[(q, ss1)])
                    else:
                        rT2 = False

            if (not rT1) and rT2 and self.verbose > 1:
                print(
                    "Shortcut for %d %d %d ---  (nTSQA=%d) : "
                    % (s, q, a, self.nTSQA[s, q, a]),
                    end="",
                )
                print(self.nTSAS_dict[(s, a)], end=" - ")
                for ss1 in self.nTSAS_dict[(s, a)]:
                    print(self.nTQSQ_dict[(q, ss1)], end=" - ")
                print()

            rT = rT1 or rT2

            if self.verbose > 2:
                print("%d >= %d : %r" % (self.nTSQA[s, q, a], self.nsamplesTQ, rT))

            if rT and self.knownTSQA[s, q, a] == 0:
                self.knownTSQA[s, q, a] = 1
                new_known = True

        if self.knownTSA[s, a] == 0 and self.nTSA[s, a] >= self.nsamplesTE:
            self.knownTSA[s, a] = 1
            new_known = True

            if self.verbose > 1:
                print("%d Known TSA[%d,%d] = " % (self.episode, s, a), end="")
                for ss1 in self.nTSAS_dict[(s, a)]:
                    self.pTSAS[s, a, ss1] = self.nTSAS[s, a, ss1] / self.nTSA[s, a]
                    print("{%d, %.3f} " % (s1, self.pTSAS[s, a, ss1]), end="")
                print()

                if self.verbose > 2:
                    for ss1 in self.nTSAS_dict[(s, a)]:
                        self.verifyTSAS(s, a, ss1)

        if self.knownTQS[q, s1] == 0 and self.nTQS[q, s1] >= self.nsamplesTQ:
            self.knownTQS[q, s1] = 1
            new_known = True

            if self.verbose > 1:
                print("%d Known TQS[%d,%d] = " % (self.episode, q, s1), end="")
                for qq1 in range(self.nQ):
                    if self.nTQSQ[q, s1, qq1] > 0:
                        pTQSQ = self.nTQSQ[q, s1, qq1] / self.nTQS[q, s1]
                        print("{%d, %.3f} " % (qq1, pTQSQ), end="")
                print()

                if self.verbose > 2:
                    for qq1 in range(self.nQ):
                        self.verifyTQSQ(q, s1, qq1)

        if self.knownRSA[s, a] == 0 and self.nRSA[s, a] >= self.nsamplesRE:
            self.knownRSA[s, a] = 1
            new_known = True

        if self.knownRQS[q, s1] == 0 and self.nRQS[q, s1] >= self.nsamplesRQ:
            self.knownRQS[q, s1] = 1
            new_known = True

        if self.verbose > 2:
            print("%d %d %d" % (self.nTSA[s, a], self.nRSA[s, a], self.nTSQA[s, q, a]))
            print("check: %r" % new_known)

        return new_known
