import gym_line_follower
import gymnasium as gym
from wrappers import RenderRerun


def main():
    env = gym.make('LineFollower-v0', gui = False, render_mode = 'rgb_array', randomize=False)
    env = RenderRerun(env, filename=None, skip_episodes=3, viewer="script")
    
    for j in range(5):
        env.reset(seed=123)

        for i in range(10):
            action = env.action_space.sample()
            obsv, reward, done, truncated, info = env.step(action)
            
            if done:
                break

    print("Done: ", done)

    #close the environment
    env.close()


if __name__ == "__main__":
    main()
