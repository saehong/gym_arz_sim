from gym.envs.registration import register

register(
	id='arz-v0',
	entry_point='gym_arz.envs:ARZ',
	)