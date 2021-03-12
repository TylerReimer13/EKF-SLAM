import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, atan2


class Plotting:
    def __init__(self):
        self.true_x, self.true_y, self.true_theta = [], [], []
        self.pred_x, self.pred_y, self.pred_theta = [], [], []
        self.pred_lm_x, self.pred_lm_y = [], []
        self.time = []

    def update(self, true_states, pred_states, time):
        self.true_x.append(true_states[0])
        self.true_y.append(true_states[1])
        self.true_theta.append(true_states[2])

        self.pred_x.append(pred_states[0])
        self.pred_y.append(pred_states[1])
        self.pred_theta.append(pred_states[2])

        self.pred_lm_x.append(pred_states[3])
        self.pred_lm_y.append(pred_states[4])

        self.time.append(time)

    def show(self):
        plt.plot(self.true_x, self.true_y, label='True')
        plt.plot(self.pred_x, self.pred_y, label='Predicted')
        plt.plot([mark.x for mark in landmarks], [mark.y for mark in landmarks], 'gX', label='True Landmarks')
        plt.plot([mean[3 + 3 * idx, 0] for idx in range(N)],
                 [mean[4 + 3 * idx, 0] for idx in range(N)], 'rX', label='Predicted Landmarks')
        plt.legend()
        plt.grid()
        plt.show()


class Landmark:
    def __init__(self, x_pos, y_pos, sig):
        self.x = x_pos
        self.y = y_pos
        self.s = sig
        self.seen = False

        self.x_hat = 0.
        self.y_hat = 0.
        self.s_hat = 0.


class Measurement:
    def __init__(self, rng, ang, j):
        self.rng = rng
        self.ang = ang
        self.id = j
        self.landmark = lm_from_id(j, landmarks)


class EKFSLAM:

    @staticmethod
    def motion(vt, wt, thet):
        # Avoid divide by zero
        if wt == 0.:
            wt += 1e-5

        # Motion without noise/errors
        theta_dot = wt * DT
        x_dot = (-vt/wt) * sin(thet) + (vt/wt) * sin(thet + wt*DT)
        y_dot = (vt/wt) * cos(thet) - (vt/wt) * cos(thet + wt*DT)
        a = np.array([x_dot, y_dot, theta_dot]).reshape(-1, 1)

        # Derivative of above motion model
        b = np.zeros((3, 3))
        b[0, 2] = (-vt/wt) * cos(thet) + (vt/wt) * cos(thet + wt*DT)
        b[1, 2] = (-vt/wt) * sin(thet) + (vt/wt) * sin(thet + wt*DT)

        return a, b

    def predict(self, prev_mean=None, prev_cov=None, ut=None, zt=None):
        Fx = np.eye(3, 3*N+3)

        f, g = self.motion(ut[0], ut[1], prev_mean[2, 0])
        mean = prev_mean + Fx.T @ f

        Gt = Fx.T @ g @ Fx + np.eye(3*N+3)
        cov = Gt @ prev_cov @ Gt.T + Fx.T @ Rt @ Fx

        for obs in zt:
            j = obs.landmark.s
            zi = np.array([obs.rng, obs.ang, obs.id]).reshape(-1, 1)
            if not obs.landmark.seen:
                mean[3+3*j, 0] = mean[0, 0] + obs.rng * cos(obs.ang + mean[2, 0])  # x
                mean[4+3*j, 0] = mean[1, 0] + obs.rng * sin(obs.ang + mean[2, 0])  # y
                mean[5+3*j, 0] = obs.landmark.s  # s
                obs.landmark.seen = True

            delt_x = mean[3+3*j, 0] - mean[0, 0]
            delt_y = mean[4+3*j, 0] - mean[1, 0]
            delt = np.array([delt_x, delt_y]).reshape(-1, 1)
            q = delt.T @ delt

            zi_hat = np.zeros((3, 1))
            zi_hat[0, 0] = np.sqrt(q)
            zi_hat[1, 0] = atan2(delt_y, delt_x) - mean[2, 0]
            zi_hat[2, 0] = obs.landmark.s

            Fxj_a = np.eye(6, 3)
            Fxj_b = np.zeros((6, 3*N))
            Fxj_b[3:, 3*j:3+3*j] = np.eye(3)
            Fxj = np.hstack((Fxj_a, Fxj_b))

            h = np.zeros((3, 6))
            h[0, 0] = -np.sqrt(q) * delt_x
            h[0, 1] = -np.sqrt(q) * delt_y
            h[0, 3] = np.sqrt(q) * delt_x
            h[0, 4] = np.sqrt(q) * delt_y
            h[1, 0] = delt_y
            h[1, 1] = -delt_x
            h[1, 2] = -q
            h[1, 3] = -delt_y
            h[1, 4] = delt_x
            h[2, 5] = q

            Hti = (1/q) * (h @ Fxj)
            Kti = cov @ Hti.T @ np.linalg.inv((Hti @ cov @ Hti.T + Qt))

            mean = mean + (Kti @ (zi-zi_hat))
            cov = (np.eye(cov.shape[0]) - Kti @ Hti) @ cov

        return mean, cov


def lm_from_id(lm_id, lm_list):
    for lmrk in lm_list:
        if lmrk.s == lm_id:
            return lmrk
    return None


def performance(pred):
    pred_dict = dict()
    pred_dict['X'] = pred[0, 0]
    pred_dict['Y'] = pred[1, 0]
    pred_dict['THETA'] = pred[2, 0]
    for n in range(N):
        pred_dict['LM_' + str(n) + ' X'] = pred[3 + 3 * n, 0]
        pred_dict['LM_' + str(n) + ' Y'] = pred[4 + 3 * n, 0]
        pred_dict['LM_' + str(n) + ' ID'] = pred[5 + 3 * n, 0]

    print('PREDICTED STATES')
    print(pred_dict)


def sensor():
    rng_noise = np.random.normal(0., Qt[0, 0])
    ang_noise = np.random.normal(0., Qt[1, 1])
    z_rng = np.sqrt((states[0] - lm.x) ** 2 + (states[1] - lm.y) ** 2) + rng_noise
    z_ang = atan2(lm.y - states[1], lm.x - states[0]) - states[2] + ang_noise
    z_j = lm.s
    z = Measurement(z_rng, z_ang, z_j)
    return z


def state_update(states, control):
    x, y, theta = states
    v, w = control

    theta_dot = w
    x_dot = v*cos(theta)
    y_dot = v*sin(theta)

    theta += (theta_dot + np.random.normal(0., Rt[2, 2])) * DT
    x += (x_dot + np.random.normal(0., Rt[0, 0])) * DT
    y += (y_dot + np.random.normal(0., Rt[1, 1])) * DT

    return np.array([x, y, theta])


if __name__ == "__main__":
    DT = .15
    t = 0.
    tf = 30.
    INF = 1000.

    lm1 = Landmark(2., 3., 0)
    lm2 = Landmark(10., 10., 1)
    lm3 = Landmark(-5., 12., 2)
    lm4 = Landmark(-10., 10., 3)
    lm5 = Landmark(0., 25., 4)
    landmarks = [lm1, lm2, lm3, lm4, lm5]
    N = len(landmarks)

    ekf = EKFSLAM()
    vis = Plotting()

    Rt = .1*np.eye(3)
    Qt = .05*np.eye(3)

    x = 0.
    y = 0.
    theta = 0.
    xr = np.array([x, y, theta]).reshape(-1, 1)
    xm = np.zeros((3 * N, 1))
    mean = np.vstack((xr, xm))
    cov = INF * np.eye(len(mean))
    cov[:3, :3] = np.zeros((3, 3))

    states = np.array([x, y, theta])
    u = np.array([2., 0.2])

    while t <= tf:
        vis.update(states.flatten().copy(), mean.flatten().copy(), t)
        zs = []
        for lm in landmarks:
            z = sensor()
            zs.append(z)

        states = state_update(states, u)
        mean, cov = ekf.predict(mean, cov, u, zs)
        t += DT

    performance(np.round(mean, 3))
    vis.show()
