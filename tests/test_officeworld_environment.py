from multiagent_rlrm.environments.office_world.ma_office import MultiAgentOfficeWorld
from multiagent_rlrm.multi_agent.agent_rl import AgentRL


def test_officeworld_reset_infos_are_keyed_by_agent_name():
    env = MultiAgentOfficeWorld(
        width=2,
        height=2,
        plants=set(),
        coffee=set(),
        letters=set(),
        walls=set(),
        plants_penalty_value=-1,
        wall_penalty_value=0,
        terminate_on_plants=False,
        terminate_hit_walls=False,
    )
    agent = AgentRL("a", env)
    agent.set_initial_position(0, 0)
    env.add_agent(agent)

    observations, infos = env.reset(seed=123)

    assert set(observations) == {"a"}
    assert set(infos) == {"a"}
