import math


class LinearWarmupInverseSqrtDecayScheduler():
    def __init__(self, optimizer, lr_initial=1e-6, lr_peak=1e-4, lr_final=2e-5, t_warmup=1000, t_decay=10000):
        assert lr_initial > 0
        assert lr_peak > 0
        assert lr_final > 0
        assert t_warmup > 0
        assert t_decay > 0

        self.optimizer = optimizer
        self.lr_initial = lr_initial
        self.lr_peak = lr_peak
        self.lr_final = lr_final
        self.t_warmup = t_warmup
        self.t_decay = t_decay

        self.t = 0
        self.lr = lr_initial

    def step(self):
        self.t += 1

        if self.t <= self.t_warmup:
            A = (self.lr_peak - self.lr_initial) / self.t_warmup
            B = self.lr_initial
            self.lr = A * self.t + B
        elif self.t > self.t_warmup and self.t <= self.t_decay:
            A = (self.lr_peak - self.lr_final) / (1 / math.sqrt(self.t_warmup) - 1 / math.sqrt(self.t_decay))
            B = self.lr_peak - A / math.sqrt(self.t_warmup)
            self.lr = A / math.sqrt(self.t) + B
        else:
            self.lr = self.lr_final

        for p in self.optimizer.param_groups:
            p["lr"] = self.lr

    def get_t(self):
        return self.t

    def get_lr(self):
        return self.lr
