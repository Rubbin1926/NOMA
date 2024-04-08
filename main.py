import gymnasium as gym
from env import NOMAenv
from stable_baselines3 import PPO

env = NOMAenv()

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    print(f"""action = {action}""")
    obs, reward, done, info = vec_env.step(action)
    print(f"""obs = {obs}""")
    print(f"""reward = {reward}""")
    # VecEnv resets automatically
    if done:
      obs = env.reset()

env.close()
