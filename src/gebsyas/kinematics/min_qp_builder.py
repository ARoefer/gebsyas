import giskardpy.symengine_wrappers as spw
import numpy as np

from giskardpy.exceptions import QPSolverException
from giskardpy.qp_solver  import QPSolver

default_bound = 1e9

class HardConstraint(object):
    def __init__(self, lower, upper, expression):
        self.lower = lower
        self.upper = upper
        self.expression = expression

class SoftConstraint(HardConstraint):
    def __init__(self, lower, upper, weight, expression):
        super(SoftConstraint, self).__init__(lower, upper, expression)
        self.weight = weight

class ControlledValue(object):
    def __init__(self, lower, upper, symbol, weight=1):
        self.lower  = lower
        self.upper  = upper
        self.symbol = symbol
        self.weight = weight


class MinimalQPBuilder(object):
    def __init__(self, hard_constraints, soft_constraints, controlled_values):
        hc = hard_constraints.items()
        sc = soft_constraints.items()
        cv = controlled_values.items()

        self.np_g = np.zeros(len(weights))
        self.H    = spw.diag(*[c.weight for _, c in cv + sc])
        self.lb   = spw.Matrix([c.lower if c.lower is not None else -default_bound for _, c in cv] + [-default_bound] * len(sc))
        self.ub   = spw.Matrix([c.upper if c.upper is not None else  default_bound for _, c in cv] + [default_bound] * len(sc))
        self.lbA  = spw.Matrix([c.lower if c.lower is not None else -default_bound for _, c in hc + sc])
        self.ubA  = spw.Matrix([c.upper if c.upper is not None else  default_bound for _, c in hc + sc])

        M_cv      = spw.Matrix([c.symbol for _, c in cv])
        self.A    = spw.Matrix([c.expr   for _, c in hc + sc]).jacobian(M_cv).row_join(spw.zeros(len(hc), len(sc)).col_join(spw.eye(len(sc))))

        self.big_ass_M = self.A.row_join(self.lbA).row_join(self.ubA).col_join(self.H.row_join(lb).row_join(ub))

        self.free_symbols = self.big_ass_M.free_symbols
        self.cython_big_ass_M = spw.speed_up(self.big_ass_M, self.free_symbols, backend=BACKEND)

        self.cv        = [c.symbol, for _, c in cv]
        self.n_cv      = len(cv)
        self.row_names = [k for k, _ in hc + sc]
        self.col_names = [k for k, _ in cv + sc]
        self.np_col_header = np.array([''] + self.col_names).reshape((1, len(self.col_names) + 1))
        self.np_row_header = np.array(self.row_names).reshape((len(self.row_names) + 1, 1))

        self.shape1    = len(self.col_names)
        self.shape2    = len(self.row_names)
        self.qp_solver = QPSolver(len(self.col_names), len(self.row_names))


    def get_cmd(self, substitutions, nWSR=None):
        np_big_ass_M = self.cython_big_ass_M(**substitutions)
        np_H   = np.array(np_big_ass_M[self.shape1:, :-2])
        np_A   = np.array(np_big_ass_M[:self.shape1, :self.shape2])
        np_lb  = np.array(np_big_ass_M[self.shape1:, -2])
        np_ub  = np.array(np_big_ass_M[self.shape1:, -1])
        np_lbA = np.array(np_big_ass_M[:self.shape1, -2])
        np_ubA = np.array(np_big_ass_M[:self.shape1, -1])
        try:
            xdot_full = self.qp_solver.solve(np_H, self.np_g, np_A, np_lb, np_ub, np_lbA, self.np_ubA, nWSR)
        except QPSolverException as e:
            print('INFEASIBLE CONFIGURATION!\n{}'.format(self._create_display_string(np_H, np_A, np_lb, np_ub, np_lbA, np_ubA)))
            raise e
        if xdot_full is None:
            return None

        return {cv: xdot_full[i] for i, cv in enumerate(self.cv)}

    def _create_display_string(self, np_H, np_A, np_lb, np_ub, np_lbA, np_ubA):
        h_str  = np.array_str(np.vstack((self.np_col_header, np.hstack((self.np_col_header.T[1:], np_H)))), precision=4)
        a_str  = np.array_str(np.vstack((self.np_col_header[:len(self.n_cv) + 1], np.hstack((self.np_row_header, np_A[:, :len(self.n_cv)])))), precision=4)
        b_str  = np.array_str(np.vstack((np.array(['', 'lb', 'ub']), np.hstack((self.np_col_header.T[1:len(self.n_cv) + 1], np_lb, np_ub)))))
        bA_str = np.array_str(np.vstack((np.array(['', 'lbA', 'ubA']), np.hstack((self.np_row_header, np_lbA, np_ubA)))))

        return 'H =\n{}\nlb, ub =\n{}\nA =\n{}\nlbA, ubA =\n{}\n'.format(h_str, b_str, a_str, bA_str)
