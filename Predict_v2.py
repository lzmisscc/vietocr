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
        return index_decode(s)
import re
def index_decode(index_encode):
    # 解码部分
    res = re.sub("\^{(.*?)}", r"<sup>\1</sup>", index_encode)
    res = re.sub("_{(.*?)}", r"<sub>\1</sub>", res)
    res = re.split("(.{0,1}[卐♡♀]{0,3})", res)
    res = list(filter(lambda x: x, res))

    print(res)

    def trans(data, split='卐', start='<b>', end='</b>'):
        res = data

        tmp_res = []
        while res:
            tmp = res.pop(0)
            tmp_ = []
            if split in tmp:
                tmp_ = [start + tmp.replace(split, ""), ]
                if res:
                    tmp = res.pop(0)
                    flag = 0
                    while split in tmp:
                        tmp_.append(tmp.replace(split, ""))
                        if res:
                            tmp = res.pop(0)
                        else:
                            flag = 1
                            break

                    tmp_[-1] += end
                    if not flag:
                        tmp_.append(tmp)
                    tmp_res += tmp_
                else:
                    tmp_res.append(start + tmp.replace(split, "") + end,)
            else:
                tmp_res.append(tmp)
        print(tmp_res)
        return tmp_res

    res = trans(res, '♡', '<i>', '</i>')
    res = trans(res, '卐', '<b>', '</b>')
    res = trans(res, '♀', '<strike>', '</stirke>')

    # 纠错部分
    # for index, label in enumerate(res):
    #     if '<b>' in label and '<i>' in label:
    #         pass
    #     elif '</b>' in label and '</i>' in label:
    #         pass
    # res = ''.join(res)
    # tmp = re.split('(<.*?>)', res)
    # tmp = list(filter(lambda x:x, tmp))
    # s = []
    # while tmp:
    #     x = tmp.pop(0)
    #     if

    print(''.join(res))
    return ''.join(res)



if __name__ == '__main__':
    import os.path as osp
    ev = Ev()
    run = ocr("config/vgg-transformer.yml").run
    table_ocr_txt_path = "../table_ocr/filter_val.txt"
    with open(table_ocr_txt_path, "r") as f:
        gt_lines = f.readlines()
    for index, line in enumerate(gt_lines):
        name, value = line.strip("\n").split("\t")
        im = Image.open(osp.join("../table_ocr/data/val", name))
        start = time.time()
        pre = run(im)
        ev.count(value, pre)
        print(f"{time.time()-start:.2f}\t{ev.socre()}")

