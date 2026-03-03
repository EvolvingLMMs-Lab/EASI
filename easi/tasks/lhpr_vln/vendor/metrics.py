"""Navigation metrics for LHPR-VLN benchmark. Vendored verbatim from LH-VLN/utils/metrics.py."""


class NavigationMetrics:
    def __init__(self):
        self.successes = []
        self.gt_steps = []
        self.gt_length = []
        self.error_length = []
        self.path_steps = []
        self.oracle_successes = []
        self.navigation_errors = []
        self.subtask_successes = []
        self.subtask_path_steps = []

    def add_sample(self,
                   success,
                   gt_step,
                   path_step,
                   oracle_success,
                   navigation_error,
                   subtask_successes,
                   subtask_path_step,
                   gt_length,
                   error_length):
        """Adds a sample of navigation results.

        Parameters:
        - success: 1 if the task is successful, 0 otherwise.
        - gt_step: The ground truth path steps of the episode.
        - path_step: The path steps of the episode.
        - oracle_success: 1 if the oracle task is successful, 0 otherwise.
        - navigation_error: The navigation error for the task.
        - subtask_successes: A list of 0s and 1s indicating subtask success or failure.
        - subtask_path_step: A list for the ground truth path steps of subtasks.
        - gt_length: Ground path length for sub-tasks,
        - error_length: Nav error length for sub-tasks:
        """
        self.successes.append(success)
        self.gt_steps.append(gt_step)
        self.path_steps.append(path_step)
        self.oracle_successes.append(oracle_success)
        self.navigation_errors.append(navigation_error)
        self.subtask_successes.append(subtask_successes)
        self.subtask_path_steps.append(subtask_path_step)
        self.gt_length.append(gt_length)
        self.error_length.append(error_length)

    def success_rate(self):
        """Calculates Success Rate (SR)."""
        return sum(self.successes) / len(self.successes) if len(self.successes) > 0 else 0

    def oracle_success_rate(self):
        """Calculates Oracle Success Rate (OSR)."""
        return sum(self.oracle_successes) / len(self.oracle_successes) if len(self.oracle_successes) > 0 else 0

    def spl(self):
        """Calculates Success Rate Penalized by Path Length (SPL)."""
        total_spl = sum(s * (gp/max(gp, p)) if p > 0 else 0 for s, gp, p in zip(self.successes, self.gt_steps, self.path_steps))
        return total_spl / len(self.successes) if len(self.successes) > 0 else 0

    def navigation_error(self):
        """Calculates average Navigation Error (NE)."""
        return sum(self.navigation_errors) / len(self.navigation_errors) if len(self.navigation_errors) > 0 else 0

    def independent_success_rate(self):
        """Calculates Independent Success Rate (ISR)."""
        subtask_counts = [len(subtasks) for subtasks in self.subtask_successes]
        total_subtasks = sum(subtask_counts)
        total_successes = sum(sum(subtasks) for subtasks in self.subtask_successes)
        return total_successes / total_subtasks if total_subtasks > 0 else 0

    def conditional_success_rate(self):
        """Calculates Conditional Success Rate (CSR)."""
        M = len(self.subtask_successes)
        if M == 0:
            return 0
        csr = 0
        for i in range(M):
            sr = 0
            N = len(self.subtask_successes[i])
            if N == 0:
                continue
            s = self.subtask_successes[i][0]
            sr += s*N
            if N == 1:
                csr += sr
                continue
            for j in self.subtask_successes[i][1:]:
                sr += j*(1+(N-1)*s)
                s = j
            csr += sr/(N**2)
        csr = csr/M
        return csr

    def conditional_path_length(self):
        """Calculates Conditional Success Rate weighted by Ground Truth Path Length (CGT)."""
        M = len(self.subtask_successes)
        if M == 0:
            return 0
        cpl = 0
        for i in range(M):
            sr = 0
            N = len(self.subtask_successes[i])
            if N == 0:
                continue
            s = self.subtask_successes[i][0]
            w = [l/sum(self.subtask_path_steps[i]) for l in self.subtask_path_steps[i]]
            sr += s*N*w[0]
            if N == 1:
                cpl += sr
                continue
            for j, wj in zip(self.subtask_successes[i][1:], w[1:]):
                sr += j*(1+(N-1)*s)*wj
                s = j
            cpl += sr/N
        cpl = cpl/M
        return cpl

    def TAR(self):
        """Calculates Target Approach Rate (TAR)."""
        tars = []
        for i in range(len(self.gt_length)):
            for j in range(len(self.gt_length[i])):
                error = self.error_length[i][j]
                gt = self.gt_length[i][j]
                if error == 0 and gt == 0:
                    tar = 1.0
                elif gt == 0:
                    tar = 0.0
                else:
                    denominator = max(error, gt)
                    if denominator > 0:
                        tar = 1 - max(error - 1, 0) / denominator
                    else:
                        tar = 1.0
                tars.append(tar)
        return sum(tars) / len(tars) if len(tars) > 0 else 0

    def compute(self):
        """Compute all metrics."""
        return {
            "success_rate": self.success_rate(),
            "oracle_success_rate": self.oracle_success_rate(),
            "spl": self.spl(),
            "navigation_error": self.navigation_error(),
            "independent_success_rate": self.independent_success_rate(),
            "conditional_success_rate": self.conditional_success_rate(),
            "conditional_path_length":self.conditional_path_length(),
            "tar":self.TAR()
        }
