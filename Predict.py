import logging
import re
from char import character
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from eval import Ev
import os.path as osp
import tqdm
import time
from distance import levenshtein
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
        self.model = self.text_r.model
        # self.model.share_memory()

    def run(self, im: Image):
        s = self.text_r.predict(im)
        return index_decode(s)


def index_decode(index_encode):
    # 解码部分
    res = re.sub("\^{(.*?)}", r"<sup>\1</sup>", index_encode)
    res = re.sub("_{(.*?)}", r"<sub>\1</sub>", res)
    res = re.split("(.{0,1}[卐♡♀]{0,3})", res)
    res = list(filter(lambda x: x, res))

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
        return tmp_res

    res = trans(res, '♡', '<i>', '</i>')
    res = trans(res, '卐', '<b>', '</b>')
    res = trans(res, '♀', '<strike>', '</stirke>')

    return ''.join(res)


if __name__ == '__main__':
    import os.path as osp
    import tqdm
    import time
    from distance import levenshtein
    ev = Ev()
    fuck = ocr("config/resnet-transformer.yml")
    # fuck = ocr("config/vgg-convseq2seq.yml")
    run = fuck.run
    table_ocr_txt_path = "../table_ocr/filter_val.txt"
    with open(table_ocr_txt_path, "r") as f:
        gt_lines = f.readlines()
    gt_lines = tqdm.tqdm(gt_lines)
    log = open(f"Predict_{time.strftime('%y-%m-%d-%H')}.csv", "w")
    result_txt = open("transformer.result", "w")
    for index, line in enumerate(gt_lines):
        name, value = line.strip("\n").split("\t")
        im = Image.open(osp.join("../table_ocr/data/val", name))
        start = time.time()
        pre = run(im)
        result_txt.write(f"{name}\t{pre}\n")
        ev.count(value, pre)
        if value != pre:
            log.write(f"{name},{pre},{value},{levenshtein}\n")
        acc = ev.socre()
        gt_lines.set_description(f"{time.time()-start:.2f},char_acc:{acc['char_acc']*100:.2f},seq_acc:{acc['seq_acc']*100:.2f},{value==pre}")