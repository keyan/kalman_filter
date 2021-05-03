import math
import unittest
from collections import namedtuple

import numpy as np

Pose = namedtuple('Pose', ['x', 'y', 'theta'])


class KalmanFilter:
    """
    An implementation of the extended Kalman filter with landmarks of known
    correspondence. Book references are to Probabilistic Robotics 2nd ed.
    """
    def __init__(self):
        self.last_pose = Pose(0, 0, 0)
        self.last_pose_timestamp = 0
        # Angular velocity.
        self.last_w = 0
        # Translational velocity.
        self.last_v = 0
        # Time delta in secs used when last pose was computed using the motion model.
        self.dt = 0

        # Define a covariance matrix which captures the relationship between variables
        # in the state. In our case the state is (x, y, theta) so we can assume all
        # variables are independent and then we just pick a reasonable variance to start
        # with (we can tune this) and we get the following diagonal matrix. When using EKF
        # the covariance is updated at every sensor fusion step.
        self.covariance = np.array([
            [5, 0, 0],
            [0, 5, 0],
            [0, 0, 2],
        ])

        # Motion model error parameters, these are somewhat arbitrary and should be tuned
        # for a particular robot. Some guidance presented in Figure 5.3 (pg. 122). In our
        # case pick small-ish values and just see what happens.
        # indices 0/1 are translational error, 2/3 are rotational error
        self.alphas = [1/8, 1/8, 1/8, 1/8]

        # Sensor model error parameters. Explained on pg. 178, equation 6.40. They are the
        # standard deviations for range, bearing, and signature (respectively) from the
        # sensor reading, defining a zero-mean Gaussian error variable.
        self.epsilons = [1/8, 1/8, 1/8]

    def update(self, static_pos_x, static_pos_y, z_measured):
        """
        Implements EKF localization with known correspondence using algorithm from
        Table 7.2. Assumes velocity motion model, so Jacobians are directly from
        the book, pg. 204.
        """
        v_t = self.last_v
        w_t = self.last_w
        theta = self.last_pose.theta

        if w_t == 0:
            # Edge case: Going straight
            # lim_{w -> 0} v/w (-cos(theta) + cos(theta + dt * w)) = -v * dt * sin(theta)
            # lim_{w -> 0} v/w (-sin(theta) + sin(theta + dt * w)) = v * dt * cos(theta)
            G = np.array([
                [1, 0, -v_t * self.dt * np.sin(theta)],
                [0, 1, v_t * self.dt * np.cos(theta)],
                [0, 0, 1],
            ])

            # For first column: see limits above G
            # For second column: some crazy wolfram alpha stuff
            #   - https://bit.ly/3eCzM2Q
            #   - Note this one is negative of function we want: https://bit.ly/3eAHLgS
            V = np.array([
                [self.dt * np.cos(theta), -.5 * v_t * self.dt ** 2 * np.sin(theta)],
                [self.dt * np.sin(theta), 0.5 * v_t * self.dt ** 2 * np.cos(theta)],
                [0, self.dt],
            ])
        else:
            # Normal case: Arc
            vt_over_wt = v_t / w_t
            cos_theta_wt_dt = math.cos(theta + (w_t * self.dt))
            sin_theta_wt_dt = math.sin(theta + (w_t * self.dt))

            G = np.array([
                [1, 0, -vt_over_wt * math.cos(theta) + vt_over_wt * cos_theta_wt_dt],
                [0, 1, -vt_over_wt * math.sin(theta) + vt_over_wt * sin_theta_wt_dt],
                [0, 0, 1],
            ])

            V = np.array([
                [
                    (-math.sin(theta) + sin_theta_wt_dt) / w_t,
                    (
                        ((v_t * (math.sin(theta) - sin_theta_wt_dt)) / w_t**2) +
                        ((v_t * cos_theta_wt_dt * self.dt) / w_t)
                    ),
                ],
                [
                    (math.cos(theta) - cos_theta_wt_dt) / w_t,
                    (
                        ((-v_t * (math.cos(theta) - cos_theta_wt_dt)) / w_t**2) +
                        ((v_t * sin_theta_wt_dt * self.dt) / w_t)
                    ),
                ],
                [0, self.dt],
            ])

        assert G.shape == (3, 3)
        assert np.isfinite(G).all()

        assert V.shape == (3, 2)
        assert np.isfinite(V).all()

        M = np.array([
            [(self.alphas[0] * (v_t**2)) + (self.alphas[1] * (w_t**2)), 0],
            [0, (self.alphas[2] * (v_t**2)) + (self.alphas[3] * (w_t**2))],
        ])
        assert M.shape == (2, 2)
        assert np.isfinite(M).all()

        # Prediction, L6-7
        # Already have u_pred from the motion model.
        u_pred = np.array([self.last_pose.x, self.last_pose.y, self.last_pose.theta]).reshape(-1, 1)
        assert u_pred.shape == (3, 1)
        assert np.isfinite(u_pred).all()

        cov_pred = (G @ self.covariance @ G.T) + (V @ M @ V.T)
        assert cov_pred.shape == (3, 3)
        assert np.isfinite(cov_pred).all()

        # Correction, L8-16
        # Incorporate the known correspondence of the observed feature.

        # Covariance of measurement noise, 7.15 and equation 6.40.
        Q = np.array([
            [self.epsilons[0]**2, 0, 0],
            [0, self.epsilons[1]**2, 0],
            [0, 0, self.epsilons[2]**2],
        ])
        assert Q.shape == (3, 3)
        assert np.isfinite(Q).all()

        # Actual positions of the landmark detected, according to the static map.
        m_x, m_y, m_s = static_pos_x, static_pos_y, 0

        pred_x, pred_y, pred_theta = u_pred.ravel()
        q = (m_x - pred_x)**2 + (m_y - pred_y)**2
        if q == 0:
            # Edge case:
            # q = 0 implies that:
            #   - (m_{j, x} - \hat{\mu}_{t, x}) = 0
            #   - and (m_{j, y} - \hat{\mu}_{t, y}) = 0

            # In the edge case calculations are bassically 0 / sqrt(0).
            # lim_{x -> 0} x / sqrt(x) = lim_{x -> 0} sqrt(x) = 0
            edge_case_x = 0
            edge_case_y = 0
            H = np.array([
                [-edge_case_x, -edge_case_y, 0],
                [edge_case_y, -edge_case_x, -1],
                [0, 0, 0],
            ])

            # arctan2(0, 0), which is undefined. It can be any value 0 through pi/2
            edge_case_atan = 0
            z_expected = np.array([0, edge_case_atan - pred_theta, m_s]).reshape(-1, 1)
        else:
            z_expected = np.array([
                math.sqrt(q), math.atan2(m_y - pred_y, m_x - pred_x) - pred_theta, m_s,
            ]).reshape(-1, 1)

            H = np.array([
                [-(m_x - pred_x) / math.sqrt(q), -(m_y - pred_y) / math.sqrt(q), 0],
                [(m_y - pred_y) / math.sqrt(q), -(m_x - pred_x) / math.sqrt(q), -1],
                [0, 0, 0],
            ])

        assert z_expected.shape == (3, 1)
        assert np.isfinite(z_expected).all()

        assert H.shape == (3, 3)
        assert np.isfinite(H).all()

        S = (H @ cov_pred @ H.T) + Q
        assert np.isfinite(S).all()

        K = cov_pred @ H.T @ np.linalg.inv(S)
        assert np.isfinite(K).all()

        # Correction using the innovation, value between actual and estimated sensor reading.
        self.covariance = (np.identity(3) - (K @ H)) @ cov_pred
        u_corrected = u_pred + (K @ (z_measured - z_expected))
        assert u_corrected.shape == (3, 1)
        assert np.isfinite(u_corrected).all()

        return u_corrected

    def motion_model(self, angular_vel, translational_vel, timestamp):
        """
        Take in velocity, update self.last_pose.
        """
        dt = timestamp - self.last_pose_timestamp
        self.dt = dt

        # Integrate the relative movement between the last pose and the current
        theta_delta = self.last_w * dt
        # to ensure no division by zero for radius calculation:
        if np.abs(self.last_w) < 0.000001:
            # straight line
            x_delta = self.last_v * dt
            y_delta = 0
        else:
            # arc of circle
            radius = self.last_v / self.last_w
            x_delta = radius * np.sin(theta_delta)
            y_delta = radius * (1.0 - np.cos(theta_delta))

        # Add to the previous to get absolute pose relative to the starting position
        theta_res = self.last_pose.theta + theta_delta
        x_res = self.last_pose.x + x_delta * np.cos(self.last_pose.theta) - y_delta * np.sin(self.last_pose.theta)
        y_res = self.last_pose.y + y_delta * np.cos(self.last_pose.theta) + x_delta * np.sin(self.last_pose.theta)

        # Update the stored last pose.
        self.last_pose = Pose(x_res, y_res, theta_res)
        self.last_pose_timestamp = timestamp
        self.last_w = angular_vel
        self.last_v = translational_vel


class TestKalmanFilter(unittest.TestCase):

    def test_move_forward(self):
        """
        If controls are just to move forward without turning and you get
        perfect observations, the corrected mean should match the motion
        model exactly.
        """
        kf = KalmanFilter()
        self.assertEqual(kf.last_pose.x, 0)
        self.assertEqual(kf.last_pose.y, 0)
        self.assertEqual(kf.last_pose.theta, 0)

        # Initial motion is ignored, so no pose change.
        kf.motion_model(angular_vel=0, translational_vel=5, timestamp=1)
        self.assertEqual(kf.last_pose.x, 0)
        self.assertEqual(kf.last_pose.y, 0)
        self.assertEqual(kf.last_pose.theta, 0)

        # 5 seconds of travel at 5m/s
        kf.motion_model(angular_vel=0, translational_vel=5, timestamp=6)
        self.assertEqual(kf.last_pose.x, 25)
        self.assertEqual(kf.last_pose.y, 0)
        self.assertEqual(kf.last_pose.theta, 0)

        # Let there be a landmark with known correspondence exactly 5m ahead.
        landmark_x, landmark_y = 30, 0

        # Now assume there was 0 error, our true pose should be equal to
        # the one produced by the motion model, assuming our sensor measurement
        # is nearly perfect.

        # Motion model error
        kf.alphas = [0, 0, 0, 0]
        # Sensor model error
        kf.epsilons = [1/10000, 1/10000, 1/10000, 1/10000]

        z_measured = np.array([
            math.sqrt((landmark_x - kf.last_pose.x)**2 + (landmark_y - kf.last_pose.y)**2),
            0,  # No angular change right now
            0,  # No signatures either
        ]).reshape(-1, 1)
        mean_corrected = kf.update(landmark_x, landmark_y, z_measured)

        self.assertEqual(mean_corrected[0], 25)
        self.assertEqual(mean_corrected[1], 0)
        self.assertEqual(mean_corrected[2], 0)


if __name__ == '__main__':
    unittest.main()
