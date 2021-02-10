import argparse
import logging
from vietocr.model.trainer import Trainer
from vietocr.tool.config import Cfg
import sys
sys.path.insert(0, './')
from char import character
logging.basicConfig(level=logging.INFO, )
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='see example at ')
    parser.add_argument('--checkpoint', help='your checkpoint')

    args = parser.parse_args()
    config_base = Cfg.load_config_from_file("config/base.yml")
    config = Cfg.load_config_from_file(args.config)
    config_base.update(config)
    config = config_base

    config['vocab'] = character
    trainer = Trainer(config, pretrained=False)


    # args.checkpoint = config.trainer["checkpoint"]
    # if args.checkpoint:
    #    trainer.load_checkpoint(args.checkpoint)
    #    logging.info(f"Load checkpoint form {args.checkpoint}....")
        
    trainer.train()

if __name__ == '__main__':
    main()
