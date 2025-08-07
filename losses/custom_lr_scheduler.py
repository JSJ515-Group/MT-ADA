def lr_poly(base_lr, i_iter, alpha=10, beta=0.75, num_steps=250000):
    if i_iter < 0:
        return base_lr
    return base_lr / ((1 + alpha * float(i_iter) / num_steps) ** (beta))



class LRScheduler:
    def __init__(self, learning_rate, warmup_learning_rate=0.0, warmup_steps=2000, num_steps=200000, alpha=10,
                 beta=0.75,
                 double_bias_lr=False, base_weight_factor=False):
        self.learning_rate = learning_rate
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.num_steps = num_steps
        self.alpha = alpha
        self.beta = beta
        self.double_bias_lr = double_bias_lr
        self.base_weight_factor = base_weight_factor

    def __call__(self, optimizer, i_iter):
        if i_iter < self.warmup_steps:
            lr_i_iter = max(i_iter - self.warmup_steps, 0)
            lr = self.warmup_learning_rate
        else:
            lr_i_iter = max(i_iter - self.warmup_steps, 0)
            lr = self.learning_rate

        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr_poly(lr, lr_i_iter, alpha=self.alpha, beta=self.beta,
                                                      num_steps=self.num_steps)
        elif len(optimizer.param_groups) == 2:
            optimizer.param_groups[0]['lr'] = lr_poly(lr, lr_i_iter, alpha=self.alpha, beta=self.beta,
                                                      num_steps=self.num_steps)
            optimizer.param_groups[1]['lr'] = (1 + float(self.double_bias_lr)) * lr_poly(lr, lr_i_iter,
                                                                                         alpha=self.alpha,
                                                                                         beta=self.beta,
                                                                                         num_steps=self.num_steps)
        elif len(optimizer.param_groups) == 4:
            optimizer.param_groups[0]['lr'] = lr_poly(lr, lr_i_iter, alpha=self.alpha, beta=self.beta,
                                                      num_steps=self.num_steps)
            optimizer.param_groups[1]['lr'] = (1 + float(self.double_bias_lr)) * lr_poly(lr, lr_i_iter,
                                                                                         alpha=self.alpha,
                                                                                         beta=self.beta,
                                                                                         num_steps=self.num_steps)
            optimizer.param_groups[2]['lr'] = self.base_weight_factor * lr_poly(lr, lr_i_iter, alpha=self.alpha,
                                                                                beta=self.beta,
                                                                                num_steps=self.num_steps)
            optimizer.param_groups[3]['lr'] = (1 + float(self.double_bias_lr)) * self.base_weight_factor * lr_poly(lr,
                                                                                                                   lr_i_iter,
                                                                                                                   alpha=self.alpha,
                                                                                                                   beta=self.beta,
                                                                                                                   num_steps=self.num_steps)
        else:
            raise RuntimeError('Wrong optimizer param groups')


    def __call__(self, optimizer, i_iter):
        if i_iter < self.warmup_steps:
            # 预热阶段：线性增长
            lr = (self.warmup_learning_rate +
                  (self.learning_rate - self.warmup_learning_rate) *
                  i_iter / self.warmup_steps)
        else:
            # 非预热阶段：多项式衰减
            lr_i_iter = i_iter - self.warmup_steps
            lr = lr_poly(self.learning_rate, lr_i_iter, alpha=self.alpha, beta=self.beta, num_steps=self.num_steps)

        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        elif len(optimizer.param_groups) == 2:
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = (1 + float(self.double_bias_lr)) * lr
        elif len(optimizer.param_groups) == 4:
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = (1 + float(self.double_bias_lr)) * lr
            optimizer.param_groups[2]['lr'] = self.base_weight_factor * lr
            optimizer.param_groups[3]['lr'] = (1 + float(self.double_bias_lr)) * self.base_weight_factor * lr
        else:
            raise RuntimeError('Wrong optimizer param groups')


    def current_lr(self, i_iter):
        if i_iter < self.warmup_steps:
            return self.warmup_learning_rate
        else:
            lr_i_iter = max(i_iter - self.warmup_steps, 0)
            return lr_poly(self.learning_rate, lr_i_iter, alpha=self.alpha, beta=self.beta, num_steps=self.num_steps)

def lr_poly(base_lr: float, i_iter: int, alpha: float = 10, beta: float = 0.75, num_steps: int = 250000) -> float:
    """
    多项式学习率衰减计算函数
    """
    if i_iter < 0:
        return base_lr
    return base_lr / ((1 + alpha * float(i_iter) / num_steps) ** beta)


class LRScheduler:
    def __init__(self, learning_rate: float, warmup_learning_rate: float = 0.0, warmup_steps: int = 2000,
                 num_steps: int = 200000, alpha: float = 10, beta: float = 0.75,
                 double_bias_lr: bool = False, base_weight_factor: float = 1.0):
        """
        学习率调度器初始化
        """
        self.learning_rate = learning_rate
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.num_steps = num_steps
        self.alpha = alpha
        self.beta = beta
        self.double_bias_lr = double_bias_lr
        self.base_weight_factor = base_weight_factor

    def get_current_lr(self, i_iter: int) -> float:
        """
        根据当前迭代步数返回当前学习率
        """
        if i_iter < self.warmup_steps:
            return self.warmup_learning_rate
        else:
            lr_i_iter = max(i_iter - self.warmup_steps, 0)
            return lr_poly(self.learning_rate, lr_i_iter, alpha=self.alpha, beta=self.beta, num_steps=self.num_steps)

    def apply_lr_to_optimizer(self, optimizer, lr: float, lr_i_iter: int):
        """
        根据优化器的参数组数量动态调整学习率
        """
        for idx, param_group in enumerate(optimizer.param_groups):
            if idx == 0:  # 第一个参数组
                param_group['lr'] = lr_poly(lr, lr_i_iter, alpha=self.alpha, beta=self.beta, num_steps=self.num_steps)
            elif idx == 1:  # 第二个参数组，可能需要双倍学习率
                param_group['lr'] = (1 + float(self.double_bias_lr)) * lr_poly(lr, lr_i_iter, alpha=self.alpha,
                                                                               beta=self.beta, num_steps=self.num_steps)
            else:  # 其他参数组，可能需要基础权重因子调整
                param_group['lr'] = self.base_weight_factor * lr_poly(lr, lr_i_iter, alpha=self.alpha, beta=self.beta,
                                                                      num_steps=self.num_steps)
                if idx == 3:  # 假设第4个参数组可能需要双倍基础权重因子
                    param_group['lr'] *= (1 + float(self.double_bias_lr))

    def __call__(self, optimizer, i_iter: int):
        """
        更新优化器的学习率
        """
        # 获取当前学习率
        if i_iter < self.warmup_steps:
            lr = self.warmup_learning_rate
            lr_i_iter = 0
        else:
            lr_i_iter = max(i_iter - self.warmup_steps, 0)
            lr = self.learning_rate

        # 应用学习率到优化器
        self.apply_lr_to_optimizer(optimizer, lr, lr_i_iter)

