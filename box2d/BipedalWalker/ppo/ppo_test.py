import torch

from ppo_agent import  BipedalWalkerPPO

car_racing_ppo = BipedalWalkerPPO(cuda=True, seed=0, torch_deterministic=True)
# car_racing_ppo.load_agent("./models/bipedal_walker_298.pt")
episodes = 12
env = car_racing_ppo.make_env(200, 0, True, 1, "random")()

for episode in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        # action, _, __, value = car_racing_ppo.get_action_and_value(
        #     torch.tensor([obs], dtype=torch.float).to(car_racing_ppo.device))
        # action = car_racing_ppo.get_action_greedy(
        #     torch.tensor([obs], dtype=torch.float).to(car_racing_ppo.device))
        action = env.action_space.sample()
        # obs, r, done, info = env.step(action.cpu().detach().numpy()[0])
        obs, r, done, info = env.step(action)
    print(f"Episode {episode + 1} ends with {info['episode']['r']} reward")
