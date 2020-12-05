import argparse
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import sys 
sys.path.insert(0, './')
from char import character
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='foo help')
    parser.add_argument('--config', required=True, help='foo help')

    args = parser.parse_args()
    config_base = Cfg.load_config_from_file("config/base.yml")
    config = Cfg.load_config_from_file(args.config)
    config_base.update(config)
    config = config_base

    config['vocab'] = character


    detector = Predictor(config)

    img = Image.open(args.img)
    s = detector.predict(img)

    print(s)

if __name__ == '__main__':
    main()
