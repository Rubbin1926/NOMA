from env import NOMAenv
import time
start_time = time.time()

env = NOMAenv()
observation, info = env.reset(seed=42)

for i in range(500):
    action = env.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    print(f"""i = {i+1}""")
    print(f"""observation = {observation}""")
    print(f"""reward = {reward}""")
    print(f"""terminated = {terminated}""")
    print(f"""truncated = {truncated}""")
    print(f"""info = {info}""")
    print("_____________________________")

    if terminated or truncated:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"程序执行时间：{elapsed_time}秒")
        env.close()

        # env.reset()


end_time = time.time()
elapsed_time = end_time - start_time
print(f"程序执行时间：{elapsed_time}秒")
env.close()


