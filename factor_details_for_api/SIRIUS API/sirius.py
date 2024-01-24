from factor_lab import Factor_lab

class Sirius(object):
    def __init__(self, config):
        self.config = config
        print('Welcome to Sirius!')
        print("Let's Play, Create, and Farm.")

    def download_data(self):
        pass

    def factor_lab(self):
        return Factor_lab(self.config)
