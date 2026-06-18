class MSELoss:

    def __call__(self, pred, target):

        return (pred - target) **2
        