import tempfile
from pathlib import Path

from multiagent_rlrm.learning_algorithms.qrmax_v2 import QRMax_v2


def test_qrmax_v2_choose_action_and_persistence(tmp_path: Path):
    algo = QRMax_v2(
        state_space_size=2,  # S x Q
        action_space_size=1,
        q_space_size=1,
        gamma=0.9,
    )

    # With equal Q-values, returns a valid action index
    action = algo.choose_action(obs=0, best=False, info={"s": 0, "q": 0})
    assert action == 0  # single action

    # Mutate environment stats and test save/load roundtrip
    algo.nTSAS[0, 0, 1] = 3
    algo.nTSA[0, 0] = 3
    algo.nTSAS_dict[(0, 0)] = {1}
    algo.pTSAS[0, 0, 1] = 1.0
    algo.knownTSA[0, 0] = 1

    fpath = tmp_path / "qrmax_env.pkl"
    algo.save_environment(fpath)

    # Reset fields to ensure load repopulates them
    algo.nTSAS.fill(0)
    algo.nTSA.fill(0)
    algo.nTSAS_dict.clear()
    algo.pTSAS.fill(0)
    algo.knownTSA.fill(0)

    algo.load_environment(fpath)

    assert algo.nTSAS[0, 0, 1] == 3
    assert algo.nTSA[0, 0] == 3
    assert algo.nTSAS_dict[(0, 0)] == {1}
    assert algo.pTSAS[0, 0, 1] == 1.0
    assert algo.knownTSA[0, 0] == 1
