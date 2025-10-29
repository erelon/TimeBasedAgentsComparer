import importlib
import yaml
import sys
import os
from datetime import datetime
from collections import defaultdict
import pandas as pd

try:
    from agents import *  # noqa: F401,F403
    from envs import *  # noqa: F401,F403
except ImportError:  # package-relative context
    from .agents import *  # noqa: F401,F403
    from .envs import *  # noqa: F401,F403

try:
    from gym_wrapper import SMDPEnvWrapper
except ImportError:
    from .gym_wrapper import SMDPEnvWrapper


def dynamic_import(class_path: str):
    module_name, class_name = class_path.rsplit('.', 1)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        # Try relative to current package if available
        if __package__:
            rel_module_name = f"{__package__}.{module_name}"
            module = importlib.import_module(rel_module_name)
        else:
            raise
    return getattr(module, class_name)


def _build_epsilon_scheduler(exp_cfg, episodes):
    sched = exp_cfg.get('epsilon_schedule') or exp_cfg.get('exploration_schedule')
    if not sched:
        return None
    start = float(sched.get('start', 0.1))
    end = float(sched.get('end', 0.01))
    decay_type = str(sched.get('decay_type', 'linear')).lower()
    decay_episodes = int(sched.get('decay_episodes', episodes))
    decay_episodes = max(1, decay_episodes)

    def linear(ep):
        progress = min(ep, decay_episodes) / decay_episodes
        return start + (end - start) * progress

    def exponential(ep):
        from math import log, exp
        if start <= 0 or end <= 0:
            return linear(ep)
        k = (log(start) - log(end)) / decay_episodes  # ensures epsilon(decay_episodes) ~= end
        return max(end, start * exp(-k * ep))

    return exponential if decay_type.startswith('exp') else linear


def train_single_agent(agent, env, episodes=200, eval_steps=20, seed=42, gym_mode=False, epsilon_scheduler=None):
    # If gym_mode we assume env is already wrapped
    if hasattr(env, 'set_seed'):
        env.set_seed(seed)
    if hasattr(env, 'reset'):
        env.reset()
    agent.seed = seed
    agent.reset()
    state = env.get_state() if not gym_mode else env.reset()[0]
    last_policy_changed_at = 0
    for episode in range(episodes):
        if epsilon_scheduler and hasattr(agent, 'exploration_rate'):
            agent.exploration_rate = epsilon_scheduler(episode)
        action = agent.act(state)
        if gym_mode:
            next_state, reward, terminated, truncated, info = env.step(action)
            time = info.get('duration', 1.0)
        else:
            time, reward = env.get_reward(agent, action)
            next_state = env.get_state()
        agent.learn(state, action, reward, next_state, time)
        if hasattr(agent, 'get_policy_changed') and agent.get_policy_changed():
            last_policy_changed_at = episode
            agent.last_policy_changed_at = episode
        state = next_state
    agent.last_policy_changed_at = last_policy_changed_at
    # Evaluation phase
    if hasattr(env, 'set_seed'):
        env.set_seed(seed + 1)
    if hasattr(env, 'reset'):
        env.reset()
    eval_rewards = []
    for _ in range(eval_steps):
        if gym_mode:
            state = env.reset()[0]  # fresh state each eval step (stateless assumption)
            action = agent.eval(state)
            next_state, reward, terminated, truncated, info = env.step(action)
        else:
            state = env.get_state()
            action = agent.eval(state)
            time, reward = env.get_reward(agent, action)
        eval_rewards.append(reward)
    return sum(eval_rewards) / len(eval_rewards)


def run_from_config(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    global_seed = config.get('global', {}).get('seed', 42)
    exp_cfg = config.get('experiment', {})
    episodes = exp_cfg.get('episodes', 5000)
    eval_steps = exp_cfg.get('eval_steps', 1000)
    epochs = exp_cfg.get('epochs', 100)
    output_csv_name = exp_cfg.get('output_csv', 'results.csv')
    gym_mode = exp_cfg.get('gym_mode', False)
    experiment_name = exp_cfg.get('name') or (config.get('environment', {}).get('name') or 'experiment').replace(' ',
                                                                                                                 '_')
    results_root = exp_cfg.get('results_root', 'results')

    epsilon_scheduler = _build_epsilon_scheduler(exp_cfg, episodes)

    os.makedirs(results_root, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(results_root, f"{experiment_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    output_csv_path = os.path.join(run_dir, output_csv_name)

    env_cfg = config['environment']
    env_class = dynamic_import(env_cfg['class'])
    env_params = env_cfg.get('params', {})
    env = env_class(env_cfg.get('name', 'Environment'), **env_params)
    if gym_mode:
        env = SMDPEnvWrapper(env)

    agents_cfg = config.get('agents', [])
    agents = []
    oracle = None
    for a in agents_cfg:
        if not a.get('enabled', True):
            continue
        agent_class = dynamic_import(a['class'])
        base_params = a.get('params', {}) or {}
        params = dict(base_params)  # copy to avoid mutating config
        if agent_class is OracleAgent and 'env_secret' not in params and not gym_mode:
            real_env = env.env if isinstance(env, SMDPEnvWrapper) else env
            if hasattr(real_env, 'secret'):
                params['env_secret'] = real_env.secret()
        agent = agent_class(name=a.get('name', agent_class.__name__), action_space=(
            env.env.get_action_space() if isinstance(env, SMDPEnvWrapper) else env.get_action_space()), **params)
        agent.seed = global_seed
        if isinstance(agent, OracleAgent):
            oracle = agent
        agents.append(agent)

    results = defaultdict(dict)
    for agent in agents:
        print(f"Agent: {agent.name}", file=sys.stderr)
        best_action_per_state = defaultdict(list)
        avg_rewards = []
        avg_last_policy_change = []
        for i in range(1, epochs + 1):
            agent.reset()
            avg_reward = train_single_agent(agent, env, episodes=episodes, eval_steps=eval_steps, seed=i,
                                            gym_mode=gym_mode, epsilon_scheduler=epsilon_scheduler)
            avg_rewards.append(avg_reward)
            avg_last_policy_change.append(getattr(agent, 'last_policy_changed_at', 0))
            if i == epochs:
                for state in agent.q_table:
                    actions = agent.q_table[state]
                    best_action = max(actions, key=actions.get)
                    best_action_per_state[state].append(best_action)
        results[agent.name] = {
            f"Average Reward over {eval_steps} steps": sum(avg_rewards) / len(avg_rewards),
            f"Avg Last Policy Change (over {epochs} runs)": sum(avg_last_policy_change) / len(avg_last_policy_change),
            "Final Epsilon": getattr(agent, 'exploration_rate', None)
        }
        print(f"Best action ratio for {agent.name}:")
        for state, actions in best_action_per_state.items():
            if oracle:
                best_action_ratio = len([i for i in actions if i == oracle.act(state)]) / len(actions)
            else:
                best_action_ratio = 0.0
            print(f"State {state}: Best Action Ratio: {best_action_ratio}")
            results[agent.name][f"State {state} Best Action Ratio"] = best_action_ratio
        print(
            f"{agent.name}: Average Reward over {eval_steps} steps: {sum(avg_rewards) / len(avg_rewards)} +- {pd.Series(avg_rewards).std()}")
        print(
            f"{agent.name}: Avg Last Policy Change (over {epochs} runs): {sum(avg_last_policy_change) / len(avg_last_policy_change)} +- {pd.Series(avg_last_policy_change).std()}")
        if hasattr(agent, 'exploration_rate'):
            print(f"{agent.name}: Final Epsilon {agent.exploration_rate}")
        print("-" * 50)
    df = pd.DataFrame(results)
    print(df.to_string())
    df.to_csv(output_csv_path, float_format="%.3f")

    with open(os.path.join(run_dir, 'config_used.yaml'), 'w') as cf:
        yaml.safe_dump(config, cf)

    return run_dir


if __name__ == '__main__':
    cfg = sys.argv[1] if len(sys.argv) > 1 else 'configs/config.yaml'
    run_from_config(cfg)
