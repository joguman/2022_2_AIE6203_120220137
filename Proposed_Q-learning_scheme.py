# import library
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

T = 0.01
Tp = T * 1/10
X_Start = 0
X_End = 2000
Bs_Position = [X_End/2, 400]
Tx_Power = 33
Noise = -104

# actionlist = list(range(60))
# actions = list(combinations(actionlist, 1))
# n_action = len(actions)
# print(actions)

actions = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,), (19,), (20,), (21,), (22,), (23,), (24,), (25,), (26,), (27,), (28,), (29,), (30,), (31,), (32,), (33,), (34,), (35,), (36,), (37,), (38,), (39,), (40,), (41,), (42,), (43,), (44,), (45,), (46,), (47,), (48,), (49,), (50,), (51,), (52,), (53,), (54,), (55,), (56,), (57,), (58,), (59,)]
n_actions = len(actions)

def action_list(action):
    if action == 0:
        period = 10
        width = 15 * math.pi / 180
    elif action == 1:
        period = 10
        width = 20 * math.pi / 180
    elif action == 2:
        period = 10
        width = 25 * math.pi / 180
    elif action == 3:
        period = 10
        width = 30 * math.pi / 180
    elif action == 4:
        period = 10
        width = 45 * math.pi / 180
    elif action == 5:
        period = 10
        width = 60 * math.pi / 180

    elif action == 6:
        period = 20
        width = 15 * math.pi / 180
    elif action == 7:
        period = 20
        width = 20 * math.pi / 180
    elif action == 8:
        period = 20
        width = 25 * math.pi / 180
    elif action == 9:
        period = 20
        width = 30 * math.pi / 180
    elif action == 10:
        period = 20
        width = 45 * math.pi / 180
    elif action == 11:
        period = 20
        width = 60 * math.pi / 180

    elif action == 12:
        period = 30
        width = 15 * math.pi / 180
    elif action == 13:
        period = 30
        width = 20 * math.pi / 180
    elif action == 14:
        period = 30
        width = 25 * math.pi / 180
    elif action == 15:
        period = 30
        width = 30 * math.pi / 180
    elif action == 16:
        period = 30
        width = 45 * math.pi / 180
    elif action == 17:
        period = 30
        width = 60 * math.pi / 180

    elif action == 18:
        period = 40
        width = 15 * math.pi / 180
    elif action == 19:
        period = 40
        width = 20 * math.pi / 180
    elif action == 20:
        period = 40
        width = 25 * math.pi / 180
    elif action == 21:
        period = 40
        width = 30 * math.pi / 180
    elif action == 22:
        period = 40
        width = 45 * math.pi / 180
    elif action == 23:
        period = 40
        width = 60 * math.pi / 180

    elif action == 24:
        period = 50
        width = 15 * math.pi / 180
    elif action == 25:
        period = 50
        width = 20 * math.pi / 180
    elif action == 26:
        period = 50
        width = 25 * math.pi / 180
    elif action == 27:
        period = 50
        width = 30 * math.pi / 180
    elif action == 28:
        period = 50
        width = 45 * math.pi / 180
    elif action == 29:
        period = 50
        width = 60 * math.pi / 180

    elif action == 30:
        period = 60
        width = 15 * math.pi / 180
    elif action == 31:
        period = 60
        width = 20 * math.pi / 180
    elif action == 32:
        period = 60
        width = 25 * math.pi / 180
    elif action == 33:
        period = 60
        width = 30 * math.pi / 180
    elif action == 34:
        period = 60
        width = 45 * math.pi / 180
    elif action == 35:
        period = 60
        width = 60 * math.pi / 180

    elif action == 36:
        period = 70
        width = 15 * math.pi / 180
    elif action == 37:
        period = 70
        width = 20 * math.pi / 180
    elif action == 38:
        period = 70
        width = 25 * math.pi / 180
    elif action == 39:
        period = 70
        width = 30 * math.pi / 180
    elif action == 40:
        period = 70
        width = 45 * math.pi / 180
    elif action == 41:
        period = 70
        width = 60 * math.pi / 180

    elif action == 42:
        period = 80
        width = 15 * math.pi / 180
    elif action == 43:
        period = 80
        width = 20 * math.pi / 180
    elif action == 44:
        period = 80
        width = 25 * math.pi / 180
    elif action == 45:
        period = 80
        width = 30 * math.pi / 180
    elif action == 46:
        period = 80
        width = 45 * math.pi / 180
    elif action == 47:
        period = 80
        width = 60 * math.pi / 180

    elif action == 48:
        period = 90
        width = 15 * math.pi / 180
    elif action == 49:
        period = 90
        width = 20 * math.pi / 180
    elif action == 50:
        period = 90
        width = 25 * math.pi / 180
    elif action == 51:
        period = 90
        width = 30 * math.pi / 180
    elif action == 52:
        period = 90
        width = 45 * math.pi / 180
    elif action == 53:
        period = 90
        width = 60 * math.pi / 180

    elif action == 54:
        period = 100
        width = 15 * math.pi / 180
    elif action == 55:
        period = 100
        width = 20 * math.pi / 180
    elif action == 56:
        period = 100
        width = 25 * math.pi / 180
    elif action == 57:
        period = 100
        width = 30 * math.pi / 180
    elif action == 58:
        period = 100
        width = 45 * math.pi / 180
    else:  # action == 59:
        period = 100
        width = 60 * math.pi / 180
    return period, width

q_state = np.zeros([4000, 2])
for i in range(0, 2000):
    q_state[i + 2000 * 0, 0] = 60
    q_state[i + 2000 * 0, 1] = 1 * i
    q_state[i + 2000 * 1, 0] = 120
    q_state[i + 2000 * 1, 1] = 1 * i

q_value = np.random.uniform(low=-1, high=0, size=(len(q_state), n_actions))
q_table = np.concatenate((q_state, q_value), axis=1)

# Env
class Env():

    def __init__(self, q_table):
        self.STATE = 0
        self.datasize = len(q_table)
        self.Q_state = q_table[:, :2]
        print("Q_state")
        print(self.Q_state)
        print("Q_state size")
        print(np.shape(self.Q_state))

    def step(self, state, action):

        V = round(state[0]/3.6, 3)
        UE_Position = state[1]

        period, width = action_list(action)

        Psi = 135 * math.pi / 180
        Gm = (math.pi * 10 ** (2.028)) / (42.64 * width + math.pi)
        Gs = 10 ** (-2.028) * Gm
        Rho = 2.028 * math.log(10) / width ** 2
        Ta = (Psi / width) ** 2 * Tp
        Eta = Ta / T

        x_list = []
        d_list = []
        PL = []
        theta = []
        G = []
        SNR = []

        for i in range(0, period + 1):
            x_list.append(UE_Position + V * T * i)
            d_list.append(math.sqrt((Bs_Position[0] - x_list[i]) ** 2 + (Bs_Position[1] - 0) ** 2))
            PL.append(3 + 40 * math.log10(d_list[i]) + 7.56 - 17.3 * math.log10(0.5) - 17.3 * math.log10(9) + 2.7 * math.log10(5.9 * 10 ** 9))  # 변경한 PL

        for i in range(0, period):

            cos2 = (d_list[0] ** 2 + d_list[i] ** 2 - (V * T * i) ** 2) / (2 * d_list[0] * d_list[i])
            if cos2 > 1:
                cos2 = 1
            theta.append(math.acos(cos2))

            if theta[i] <= width:
                G.append(Gm * math.exp(-1 * Rho * theta[i] ** 2))
            else:
                G.append(Gs)

            SNR.append(Tx_Power + G[i] - PL[i] - Noise)

        t = [(1 - Eta) * math.log2(1 + 10 ** (SNR[0] / 10))]
        count = 0
        for i in range(1, period):

            if x_list[i] < X_End:
                t.append(math.log2(1 + 10 ** (SNR[i] / 10)))
                count += 1
            else:
                break

        sumthroughput = sum(t)
        reward = sumthroughput

        UE_Position = [x_list[period], 0]
        UE_Position = UE_Position[0]

        X = round(UE_Position)

        if X > 1999:
            X = 0

        return X, reward


# parameter
learning_rate = 0.99
discount_rate = 0.99
episodes = 1000000
epsilon = 0.05

env = Env(q_table)

rList_60 = []
rList_120 = []

# RL training
for episode in tqdm(range(episodes)):

    # V=60
    state = [60, random.randrange(0, 100)]
    rAll_60 = 0
    while (True):

        if np.random.uniform() < epsilon:
            action = np.random.randint(len(actions))
        else:
            action = np.argmax(q_table[(q_table[:, 0] == state[0]) & (q_table[:, 1] == state[1]), 2:])

        X, reward = env.step(state, action)
        next_state = [60, X]

        q_table[(q_table[:, 0] == state[0]) & (q_table[:, 1] == state[1]), action + 2] = \
            (1 - learning_rate) * q_table[(q_table[:, 0] == state[0]) & (q_table[:, 1] == state[1]), action + 2] \
            + learning_rate * (reward + discount_rate * np.max(q_table[(q_table[:, 0] == next_state[0]) & (q_table[:, 1] == next_state[1]), 2:]))

        if next_state[1] == 0:
            break

        rAll_60 += reward
        state = next_state
    rList_60.append(rAll_60)

    # V=120
    state = [120, random.randrange(0, 100)]
    rAll_120 = 0
    while (True):

        if np.random.uniform() < epsilon:
            action = np.random.randint(len(actions))
        else:
            action = np.argmax(q_table[(q_table[:, 0] == state[0]) & (q_table[:, 1] == state[1]), 2:])
        X, reward = env.step(state, action)
        next_state = [120, X]

        q_table[(q_table[:, 0] == state[0]) & (q_table[:, 1] == state[1]), action + 2] = (1 - learning_rate) * q_table[(q_table[:, 0] == state[0]) & (q_table[:, 1] == state[1]), action + 2] + learning_rate * (reward + discount_rate * np.max(q_table[(q_table[:, 0] == next_state[0]) & (q_table[:, 1] == next_state[1]), 2:]))

        if next_state[1] == 0:
            break

        rAll_120 += reward
        state = next_state
    rList_120.append(rAll_120)

np.savetxt('Q_table.csv',q_table,delimiter=",")

plt.figure(figsize=(6,4))
plt.plot(range(len(rList_60)), rList_60)
plt.title('60km/h')
plt.xlabel('episode')
plt.ylabel('reward')
plt.show()

plt.figure(figsize=(6,4))
plt.plot(range(len(rList_120)), rList_120)
plt.title('120km/h')
plt.xlabel('episode')
plt.ylabel('reward')
plt.show()