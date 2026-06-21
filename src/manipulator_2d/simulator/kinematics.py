import numpy as np

from manipulator_2d.constants import LINK_LENGTHS


class ForwardKinematics:
    def __init__(self):
        self.L1, self.L2, self.L3 = LINK_LENGTHS

    def jacobian_world(self, q, base_rot=0):
        q1, q2, q3 = q
        cr, sr = np.cos(base_rot), np.sin(base_rot)
        c1, s1 = np.cos(q1), np.sin(q1)
        c12, s12 = np.cos(q1 + q2), np.sin(q1 + q2)
        c123, s123 = np.cos(q1 + q2 + q3), np.sin(q1 + q2 + q3)

        j11 = (-self.L1 * s1 - self.L2 * s12 - self.L3 * s123) * cr - (
            self.L1 * c1 + self.L2 * c12 + self.L3 * c123
        ) * sr
        j21 = (-self.L1 * s1 - self.L2 * s12 - self.L3 * s123) * sr + (
            self.L1 * c1 + self.L2 * c12 + self.L3 * c123
        ) * cr
        j12 = (-self.L2 * s12 - self.L3 * s123) * cr - (self.L2 * c12 + self.L3 * c123) * sr
        j22 = (-self.L2 * s12 - self.L3 * s123) * sr + (self.L2 * c12 + self.L3 * c123) * cr
        j13 = (-self.L3 * s123) * cr - (self.L3 * c123) * sr
        j23 = (-self.L3 * s123) * sr + (self.L3 * c123) * cr

        return np.array([[j11, j12, j13], [j21, j22, j23]])
