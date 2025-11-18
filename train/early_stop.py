class EarlyStopping:
    def __init__(self, patience=10, check_freq=1, delta=0):
        """
        Args:
            patience (int): 容忍多少個 epoch 無改善
            delta (float): 最小改善量
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.check_freq = check_freq

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            
        else:
            self.counter += self.check_freq
            if self.counter >= self.patience:
                self.early_stop = True
