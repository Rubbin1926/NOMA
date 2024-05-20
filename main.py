import gymnasium as gym
from env import NOMAenv
from stable_baselines3 import PPO

env = NOMAenv()


model = PPO("MultiInputPolicy", env, verbose=2, learning_rate=1e-6, seed=42, batch_size=32)
model.learn(total_timesteps=200)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(20):
    action, _states = model.predict(obs, deterministic=True)
    print("action = ", action)
    obs, reward, done, info = vec_env.step(action)
    print("obs = ", obs)
    print("reward = ", reward)

    # VecEnv resets automatically
    if done:
        print("_____________")
        obs = vec_env.reset()

env.close()
