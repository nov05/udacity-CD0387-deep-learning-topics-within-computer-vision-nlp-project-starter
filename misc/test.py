import argparse

class Config:
    def __init__(self):
        self.debug = False

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--debug', type=bool, default=False)

    config = Config()
    for key, value in vars(parser.parse_args()).items():
        setattr(config, key, value)
    
    print(config.__dict__)

## $ python misc\test.py