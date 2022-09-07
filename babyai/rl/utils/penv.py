from multiprocessing import Process, Pipe
import gym

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        elif cmd == "setup_subgoal":
            is_setup = env.setup_subgoal(data)
            conn.send(is_setup)
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        self.processes = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()
            self.processes.append(p)

    def setup_subgoal(self, subgoal_indices):
        for local, subgoal_indice in zip(self.locals, subgoal_indices[1:]):
            local.send(("setup_subgoal", subgoal_indice))
        results = [self.envs[0].setup_subgoal(subgoal_indices[0])] + [local.recv() for local in self.locals]
        return results

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return results

    # Input:
    #   active_env_indices: the indices of parallely running environments, each of which has not completed
    #   its high-level action determined by the high-level policy agent. This variable is only used when
    #   training a hierarchical reinforcement learning agent. The indices are stored in the ascending order.
    #   So, the environment running in the parent process, the self.envs[0], is active if
    #   active_env_indices[0] is 0.
    def step(self, actions, active_env_indices=None):
        results = None
        if active_env_indices is None:
            for local, action in zip(self.locals, actions[1:]):
                local.send(("step", action))
            obs, reward, done, info = self.envs[0].step(actions[0])
            if done:
                obs = self.envs[0].reset()
            results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        else:
            is_env0_active = (active_env_indices[0]==0)
            locals = [self.locals[i-1] for i in (active_env_indices[1:] if is_env0_active else active_env_indices)]
            local_actions = actions[1:] if is_env0_active else actions
            assert len(locals)==len(local_actions)

            for local, action in zip(locals, local_actions):
                local.send(("step", action))

            if is_env0_active:
                obs, reward, done, info = self.envs[0].step(actions[0])
                if done:
                    obs = self.envs[0].reset()
                results = zip(*[(obs, reward, done, info)] + [local.recv() for local in locals])
            else:
                results = zip(*[local.recv() for local in locals])            
        return results

    def render(self):
        raise NotImplementedError

    def __del__(self):
        for p in self.processes:
            if p.is_alive():
                p.terminate()
