import numpy as np

class ForwardKinematics:
    def __init__(self):
        self.L1, self.L2, self.L3 = 0.18, 0.15, 0.12

    def jacobian_world(self, q, base_rot=0):
        q1, q2, q3 = q
        cr, sr = np.cos(base_rot), np.sin(base_rot)
        c1, s1 = np.cos(q1), np.sin(q1)
        c12, s12 = np.cos(q1 + q2), np.sin(q1 + q2)
        c123, s123 = np.cos(q1 + q2 + q3), np.sin(q1 + q2 + q3)

        J = np.zeros((2, 3))
        J[0, 0] = (-self.L1*s1 - self.L2*s12 - self.L3*s123)*cr - (self.L1*c1 + self.L2*c12 + self.L3*c123)*sr
        J[1, 0] = (-self.L1*s1 - self.L2*s12 - self.L3*s123)*sr + (self.L1*c1 + self.L2*c12 + self.L3*c123)*cr
        J[0, 1] = (-self.L2*s12 - self.L3*s123)*cr - (self.L2*c12 + self.L3*c123)*sr
        J[1, 1] = (-self.L2*s12 - self.L3*s123)*sr + (self.L2*c12 + self.L3*c123)*cr
        J[0, 2] = (-self.L3*s123)*cr - (self.L3*c123)*sr
        J[1, 2] = (-self.L3*s123)*sr + (self.L3*c123)*cr
        return J