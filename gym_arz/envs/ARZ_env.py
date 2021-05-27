import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.toy_text import discrete
import numpy as np
import ipdb

from gym_arz.envs.ARZ_Utils import *

import random


DISCRETE = False



class ARZ(gym.Env):


	def __init__(self, sett, cont_sett):


		# Set params
		self.cont_sett = cont_sett
		self.sett = sett
		self.T = sett['T']
		self.dt = sett['dt']
		self.dx = sett['dx']
		self.vm = sett['vm']
		self.rm = sett['rm']
		qm = sett['qm']
		self.L = sett['L']
		self.tau = sett['tau']



		##########################
		# DETMINISTIC ENV.
		##########################
		print('Determinstic Env.')
		self.vs = sett['vs']
		self.vs_desired = sett['vs_desired']
		self.rs = sett['rs']
		self.rs_desired = sett['rs_desired']
		self.qs = sett['qs']
		self.qs_desired = sett['qs_desired']
		self.ps = sett['ps']



		##########################
		# STOCHASTIC ENV.
		##########################
		# r1 = random.randint(0, 2)
		# #ipdb.set_trace()
		# if r1 == 0:
		# 	print('rs=0.115 case')
		# 	self.rs = 0.115
		# 	self.vs = Veq(self.vm, self.rm, self.rs)
		# 	self.qs = self.rs * self.vs

		# 	self.vs_desired = self.vs
		# 	self.rs_desired = self.rs
		# 	self.qs_desired = self.qs
		# 	# ipdb.set_trace()

		# elif r1 == 1:
		# 	print('rs=0.12 case')
		# 	self.rs = 0.12
		# 	self.vs = Veq(self.vm, self.rm, self.rs)#10
		# 	self.qs = self.rs * self.vs

		# 	self.vs_desired = self.vs
		# 	self.rs_desired = self.rs
		# 	self.qs_desired = self.qs

		# 	# ipdb.set_trace()
		# else:
		# 	print('rs=0.125 case')
		# 	self.rs = 0.125
		# 	self.vs = Veq(self.vm, self.rm, self.rs)
		# 	self.qs = self.rs * self.vs

		# 	self.vs_desired = self.vs
		# 	self.rs_desired = self.rs
		# 	self.qs_desired = self.qs


		##########################
		# STOCHASTIC Validation.
		##########################
		# r1 = random.randint(0, 2)
		
		# rs = 0.115
		# print('rs=0.115 case')
		# self.rs = 0.115
		# self.vs = Veq(self.vm, self.rm, self.rs)
		# self.qs = self.rs * self.vs

		# self.vs_desired = self.vs
		# self.rs_desired = self.rs
		# self.qs_desired = self.qs

		# rs = 0.12
		# print('rs=0.12 case')
		# self.rs = 0.12
		# self.vs = Veq(self.vm, self.rm, self.rs)#10
		# self.qs = self.rs * self.vs

		# self.vs_desired = self.vs
		# self.rs_desired = self.rs
		# self.qs_desired = self.qs


		# rs = 0.125
		# print('rs=0.125 case')
		# self.rs = 0.125
		# self.vs = Veq(self.vm, self.rm, self.rs)
		# self.qs = self.rs * self.vs

		# self.vs_desired = self.vs
		# self.rs_desired = self.rs
		# self.qs_desired = self.qs


		##########################
		##########################


		self.T = sett['T']

		x = np.arange(0,self.L+self.dx,self.dx)
		self.M = len(x)


		self.discrete = DISCRETE
		self.t = 0

		# Define input space
		self.qs = self.qs
		self.qs_input = np.linspace(self.qs/2,2*self.qs,40) #np.arange(0,qm,0.01)

		# Define variables
		self.r = np.zeros([self.M,1])
		self.y = np.zeros([self.M,1])

		# characteristics
		lambda_1 = self.vs
		lambda_2 = self.vs - self.rs * self.vm/self.rm

		# Initial condition
		print('Initial condition of rs: {}'.format(self.rs))

		self.r = self.rs * np.transpose(np.sin(3 * x / self.L * np.pi ) * 0.1 + np.ones([1,self.M]))
		self.y = self.qs * np.ones([self.M,1]) - self.vm * self.r + self.vm / self.rm * (self.r)**(2)

		# ipdb.set_trace()
		# Boundary condition // Not used in the __init__
		# self.r[0] = self.r[1]
		# self.y[0] = self.qs - self.r[0]*Veq(self.vm, self.rm, self.r[0])

		# Ghost condition
		# self.r[-1] = self.r[-2] # commented out, 2020.04.23
		# self.y[-1] = self.y[-2] # commented out, 2020.04.23


		self.v = self.y/self.r + Veq(self.vm, self.rm, self.r)

		self.info = dict()
		self.info['V'] = self.v


	@property
	def observation_space(self):
		return spaces.Box(low=-10, high=10, shape=(2 * self.M,), dtype=np.float32)
		# return spaces.Box(low=-2, high=2*qm, shape=(2 * M,), dtype=np.float32)

	@property
	def action_space(self):
		if self.discrete:
			return spaces.Discrete(20)
		else:
			# Specify the input shape.
			return spaces.Box(dtype=np.float32, low = self.qs * 0.8, high = 1.2 * self.qs, shape=(2,))

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):



		dt = self.dt
		dx = self.dx


		self.t += dt

		if self.discrete:
			qs_input = self.qs_input[action]
		else:
			qs_input = action
			

			# Single Input ------------------------------------------------------------------
			qs_input = np.clip(qs_input, a_min=self.action_space.low, a_max=self.action_space.high)[0]	
			# ------------------------------------------------------------------

			# Multiple Input ------------------------------------------------------------------
			# qs_input = np.clip(qs_input, a_min=self.action_space.low, a_max=self.action_space.high)
			# q_inlet_input = qs_input[0]
			# q_outlet_input = qs_input[1]
			# ------------------------------------------------------------------

		# Control inlet boundary input (single-input)
		# self.q_inlet = qs_input

		# Control inlet boundary input (Multi-input)
		# self.q_inlet = q_inlet_input

		# Fixed inlet boundary input
		self.q_inlet = self.qs

		# time-varying inlet boundary input
		#self.q_inlet = self.qs * np.transpose(np.sin(4 * self.t / self.T * np.pi) * 0.1 + 1)

		# Random inlet boundary input
		# self.q_inlet = self.qs * (0.1 * np.random.randn() + 1) ## Gaussian dist.
		# self.q_inlet = self.qs * (2 * np.random.uniform(-0.1,0.1) + 1) ## Uniform dist.

		# Boundary conditions
		self.r[0] = self.r[1]
		# self.y[0] = self.qs - self.r[0] * Veq(self.vm, self.rm, self.r[0])
		self.y[0] = self.q_inlet - self.r[0] * Veq(self.vm, self.rm, self.r[0]) # <-- inlet boundary

		# Ghost condition
		# M-1 means boundary
		self.r[self.M-1] = self.r[self.M-2]
		# self.y[self.M-1] = self.y[self.M-2] # commented out, 2020.04.23

		# PDE control part.------------------------------------------------------------------
		#print(action)

		# Control outlet boundary input (Multi-input)
		# self.y[self.M-1] = q_outlet_input - self.r[self.M-1]* Veq(self.vm, self.rm, self.r[self.M-1])
		
		# Control outlet boundary input
		self.y[self.M-1] = qs_input - self.r[self.M-1]* Veq(self.vm, self.rm, self.r[self.M-1])
		
		# Fixed outlet boundary input
		# self.y[self.M-1] = self.qs - self.r[self.M-1]* Veq(self.vm, self.rm, self.r[self.M-1])
		
		# ------------------------------------------------------------------

		for j in range(1,self.M-1) :

			r_pmid = 1/2 * (self.r[j+1] + self.r[j]) - dt/(2 * dx) * ( F_r(self.vm, self.rm, self.r[j+1], self.y[j+1]) - F_r(self.vm, self.rm, self.r[j], self.y[j]) )

			y_pmid = 1/2 * (self.y[j+1] + self.y[j]) - dt/(2 * dx) * ( F_y(self.vm, self.rm, self.r[j+1], self.y[j+1]) - F_y(self.vm, self.rm, self.r[j], self.y[j])) - 1/4 * dt / self.tau * (self.y[j+1]+self.y[j])

			r_mmid = 1/2 * (self.r[j-1] + self.r[j]) - dt/(2 * dx) * ( F_r(self.vm, self.rm, self.r[j], self.y[j]) - F_r(self.vm, self.rm, self.r[j-1], self.y[j-1]))

			y_mmid = 1/2 * (self.y[j-1] + self.y[j]) - dt/(2 * dx) * ( F_y(self.vm, self.rm, self.r[j], self.y[j]) - F_y(self.vm, self.rm, self.r[j-1], self.y[j-1])) - 1/4 * dt / self.tau * (self.y[j-1]+self.y[j])

			self.r[j] = self.r[j] - dt/dx * (F_r(self.vm, self.rm, r_pmid, y_pmid) - F_r(self.vm, self.rm, r_mmid, y_mmid))
			self.y[j] = self.y[j] - dt/dx * (F_y(self.vm, self.rm, r_pmid, y_pmid) - F_y(self.vm, self.rm, r_mmid, y_mmid)) - 1/2 * dt/self.tau * (y_pmid + y_mmid)

		# Calculate Velocity
		self.v = self.y/self.r + Veq(self.vm, self.rm, self.r)


		#print(self.r)

		# Reward
		v_desired = self.vs_desired #vs
		r_desired = self.rs_desired #rs
		# reward = -(np.linalg.norm(self.v-v_desired, ord=2)/(v_desired) + np.linalg.norm(self.r-r_desired, ord=2)/(r_desired))
		reward = -(np.linalg.norm(self.v-v_desired, ord=None)/(v_desired) + np.linalg.norm(self.r-r_desired, ord=None)/(r_desired))

		#print(reward)
		#print(reward)

		# Done mask
		is_done = False
		if all(self.r - r_desired == 0) and all(self.v - v_desired == 0) :
			is_done = True

		# if self.t >= self.T/self.dt :
		if self.t >= self.T/self.dt :
			print("Time over..")
			is_done = True

		# Return
		#import ipdb; ipdb.set_trace() /// debuging
		return np.reshape(np.concatenate(((self.r-self.rs_desired)/self.rs_desired, (self.v-self.vs_desired)/self.vs_desired)), -1), reward, is_done, self.info

	def render(self):
		pass


	def reset(self):
		self.__init__(self.sett, self.cont_sett)
		return np.reshape(np.concatenate(((self.r-self.rs_desired)/self.rs_desired, (self.v-self.vs_desired)/self.vs_desired)), -1)
