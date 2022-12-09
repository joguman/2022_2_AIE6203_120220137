# import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
import random

T = 0.01
Tp = T * 1/10
X_Start = 0
X_End = 2000
Bs_Position = [X_End/2, 400]
Tx_Power = 33
Noise = -104

def step(state, action):

    V = round(state[0]/3.6,3)
    ULS_Position = state[1]

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
        x_list.append(ULS_Position + V * T * i)
        d_list.append(math.sqrt((Bs_Position[0] - x_list[i]) ** 2 + (Bs_Position[1] - 0) ** 2))
        PL.append(3 + 40 * math.log10(d_list[i]) + 7.56 - 17.3 * math.log10(0.5) - 17.3 * math.log10(9) + 2.7 * math.log10(5.9 * 10**9))  # 변경한 PL

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

        t = [10**7 * (1 - Eta) * math.log2(1 + 10 ** (SNR[0] / 10))]

    for i in range(1, period):

        if x_list[i] < X_End:
            t.append(10**7 * math.log2(1 + 10 ** (SNR[i] / 10)))
        else:
            break

    SumOfThrouput = sum(t)
    reward = SumOfThrouput / 10**9 # G(기가)로 변환

    ULS_Position = [x_list[period], 0]
    ULS_Position = ULS_Position[0]
    X = round(ULS_Position)

    if X > 1999:
        X = 0

    return reward, X

actions = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,),
           (17,), (18,), (19,), (20,), (21,), (22,), (23,), (24,), (25,), (26,), (27,), (28,), (29,), (30,), (31,),
           (32,), (33,), (34,), (35,), (36,), (37,), (38,), (39,), (40,), (41,), (42,), (43,), (44,), (45,), (46,),
           (47,), (48,), (49,), (50,), (51,), (52,), (53,), (54,), (55,), (56,), (57,), (58,), (59,)]
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

LS_list = []
Proposed_list = []
Random_list = []

# V = 60
V = 60

filename = 'Q_table_LS.csv'
q_table = pd.read_csv(filename, header=None)
q_table = q_table.values

X = 0
rList = []
LS_x_sum = [0]
LS_y_sum = [0]
LS_sum_sum = 0

for i in range(0, 1000):
    state = [V, X]
    action = np.argmax(q_table[(q_table[:, 0] == state[0]) & (q_table[:, 1] == state[1]), 2:])
    reward, X = step(state, action)
    LS_sum_sum += reward

    period, width = action_list(action)

    if X != 0:
        LS_x_sum.append(X)
        LS_y_sum.append(LS_sum_sum)

    next = X
    rList.append(reward)

    if next == 0:
        LS_x_sum.append(2000)
        LS_y_sum.append(LS_sum_sum)
        break

LS_60_list = rList
print(f'V={V}km/h Local Search {sum(rList)}')
LS_list.append(sum(rList))

filename = 'Q_table.csv'
q_table = pd.read_csv(filename, header=None)
q_table = q_table.values

X = 0
rList = []
Proposed_x_avg = [0]
Proposed_y_avg = [0]
Proposed_sum_avg = 0
for i in range(0, 1000):
    state = [V, X]
    action = np.argmax(q_table[(q_table[:, 0] == state[0]) & (q_table[:, 1] == state[1]), 2:])
    reward, X = step(state, action)
    Proposed_sum_avg += reward

    period, width = action_list(action)

    if X != 0:
        Proposed_x_avg.append(X)
        Proposed_y_avg.append(Proposed_sum_avg)

    next = X
    rList.append(reward)

    if next == 0:
        Proposed_x_avg.append(2000)
        Proposed_y_avg.append(Proposed_sum_avg)
        break

Proposed_60_list = rList
print(f'V={V}km/h Proposed scheme {sum(rList)}')
Proposed_list.append(sum(rList))

# by random
X = 0
rList = []
Random_x = [0]
Random_y = [0]
Random_sum = 0

for i in range(0, 1000):
    state = [V, X]
    action = np.random.randint(len(actions))
    reward, X = step(state, action)

    Random_sum += reward

    period, width = action_list(action)

    if X != 0:
        Random_x.append(X)
        Random_y.append(Random_sum)

    next = X
    rList.append(reward)

    if next == 0:
        Random_x.append(2000)
        Random_y.append(Random_sum)
        break

Random_60_list = rList
print(f'V={V}km/h Random action {sum(rList)}')
Random_list.append(sum(rList))

plt.plot(Random_x, Random_y, 'k:', marker='s', markevery=2, label='Random Action scheme')
plt.plot(LS_x_sum, LS_y_sum, 'b--', marker='o', markevery=2, label='Proposed Local search scheme')
plt.plot(Proposed_x_avg, Proposed_y_avg, 'r-', marker='*', markevery=2, label='Proposed Q-learning scheme')
plt.xticks(np.arange(0,2001,200))
plt.xlim(0,2000)
plt.yticks(np.arange(0,801,100))
plt.ylim(0,803)
plt.tick_params(axis='x', direction='in')
plt.tick_params(axis='y', direction='in')
plt.xlabel('Distance From Start Position (m)')
plt.ylabel('Sum rate (Gbps)')
plt.legend(loc='upper left', frameon=True, edgecolor='black')
plt.grid(True, linestyle='--')
plt.savefig(f'./V={V}.png')
plt.show()

# V = 120
V = 120

filename = 'Q_table_LS.csv'
q_table = pd.read_csv(filename, header=None)
q_table = q_table.values

X = 0
rList = []
LS_x_sum = [0]
LS_y_sum = [0]
LS_sum_sum = 0

for i in range(0, 1000):
    state = [V, X]
    action = np.argmax(q_table[(q_table[:, 0] == state[0]) & (q_table[:, 1] == state[1]), 2:])
    reward, X = step(state, action)
    LS_sum_sum += reward

    period, width = action_list(action)

    if X != 0:
        LS_x_sum.append(X)
        LS_y_sum.append(LS_sum_sum)

    next = X
    rList.append(reward)

    if next == 0:
        LS_x_sum.append(2000)
        LS_y_sum.append(LS_sum_sum)
        break

LS_120_list = rList
print(f'V={V}km/h Local Search {sum(rList)}')
LS_list.append(sum(rList))

filename = 'Q_table.csv'
q_table = pd.read_csv(filename, header=None)
q_table = q_table.values

X = 0
rList = []
Proposed_x_avg = [0]
Proposed_y_avg = [0]
Proposed_sum_avg = 0

for i in range(0, 1000):
    state = [V, X]
    action = np.argmax(q_table[(q_table[:, 0] == state[0]) & (q_table[:, 1] == state[1]), 2:])
    reward, X = step(state, action)
    Proposed_sum_avg += reward

    period, width = action_list(action)

    if X != 0:
        Proposed_x_avg.append(X)
        Proposed_y_avg.append(Proposed_sum_avg)

    next = X
    rList.append(reward)

    if next == 0:
        Proposed_x_avg.append(2000)
        Proposed_y_avg.append(Proposed_sum_avg)
        break

Proposed_120_list = rList
print(f'V={V}km/h Proposed scheme {sum(rList)}')
Proposed_list.append(sum(rList))

# by random
X = 0
rList = []
Random_x = [0]
Random_y = [0]
Random_sum = 0

for i in range(0, 1000):
    state = [V, X]
    action = np.random.randint(len(actions))
    reward, X = step(state, action)

    Random_sum += reward

    period, width = action_list(action)

    if X != 0:
        Random_x.append(X)
        Random_y.append(Random_sum)

    next = X
    rList.append(reward)

    if next == 0:
        Random_x.append(2000)
        Random_y.append(Random_sum)
        break

Random_120_list = rList
print(f'V={V}km/h Random action {sum(rList)}')
Random_list.append(sum(rList))

plt.plot(Random_x, Random_y, 'k:', marker='s', markevery=2, label='Random Action scheme')
plt.plot(LS_x_sum, LS_y_sum, 'b--', marker='o', markevery=2, label='Proposed Local search scheme')
plt.plot(Proposed_x_avg, Proposed_y_avg, 'r-', marker='*', markevery=2, label='Proposed Q-learning scheme')
plt.xticks(np.arange(0,2001,200))
plt.xlim(0,2000)
plt.yticks(np.arange(0,801,100))
plt.ylim(0,803)
plt.tick_params(axis='x', direction='in')
plt.tick_params(axis='y', direction='in')
plt.xlabel('Distance From Start Position (m)')
plt.ylabel('Sum rate (Gbps)')
plt.legend(loc='upper left', frameon=True, edgecolor='black')
plt.grid(True, linestyle='--')
plt.savefig(f'./V={V}.png')
plt.show()

Velocity = ['60 (km/h)', '120 (km/h)']
x = pd.Series([1,2])

plt.xticks(x + 0.2, Velocity)
plt.bar(x,Random_list,color='k', width=0.2, label='Random Action scheme')
plt.bar(x+0.2,LS_list,color='b', width=0.2, label='Proposed Local search scheme')
plt.bar(x+0.4,Proposed_list,color='r', width=0.2, label='Proposed Q-learning scheme')
plt.legend(frameon=True, edgecolor='black')
plt.xlabel("Velocity (km/h)")
plt.ylabel("Cumulative Sum Rate (Gbps)")
plt.tick_params(axis='x', direction='in')
plt.tick_params(axis='y', direction='in')
plt.yticks(np.arange(0,801,100))
plt.ylim(0,803)
plt.grid(True, linestyle='--')
plt.savefig(f'.compare.png')
plt.show()

print("Proposed Local seacrh scheme list :", LS_list)
print("Proposed Q-learing scheme list :", Proposed_list)
print("Random Action scheme list :", Random_list)