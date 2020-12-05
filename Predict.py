from char import character
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import tqdm
import time
from string_distance.edit_distance import levenshtein
from eval import Ev

class ocr:
    def __init__(self, config, ) -> None:
        super(ocr, self).__init__()
        self.config = config
        config_base = Cfg.load_config_from_file("config/base.yml")
        config = Cfg.load_config_from_file(self.config)
        config_base.update(config)
        config = config_base
        config['vocab'] = character
        self.text_r = Predictor(config)

    def run(self, im: Image):
        s = self.text_r.predict(im)
        return s


if __name__ == '__main__':
    ev = Ev()
    run = ocr("config/vgg-seq2seq.yml").run
    table_ocr_txt_path = "../table_ocr/abs_val.txt"
    with open(table_ocr_txt_path, "r") as f:
        gt_lines = f.readlines()
    for index, line in enumerate(gt_lines):
        name, value = line.strip("\n").split("\t")
        im = Image.open(name)
        start = time.time()
        pre = run(im)
        ev.count(value, pre)
        print(f"{time.time()-start:.2f}\t{ev.socre()}")
