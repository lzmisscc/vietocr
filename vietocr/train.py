import argparse

from vietocr.model.trainer import Trainer
from vietocr.tool.config import Cfg
import sys
sys.path.insert(0, './')
from char import character

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='see example at ')
    parser.add_argument('--checkpoint', required=False, help='your checkpoint')

    args = parser.parse_args()
    config_base = Cfg.load_config_from_file("config/base.yml")
    config = Cfg.load_config_from_file(args.config)
    config_base.update(config)
    config = config_base

    # with open("table_ocr/dict.txt", "r") as f:
    #     t = []
    #     for i in f.readlines():
    #         t.append(i.strip('\n'))
    #     character = set(t)
    #     character.update('\u2028')
    # character = '\\̂●♡3\'✔Wκ=§ßΠP_⁎˂:ł∥w\u2028.h\ufeff9I〉⋆8║‡\xadY)Δ}ψ‒Ȧ̃∧ηıφ2“≥u◊→°÷С∼˃̧6〈/ [B̊€‰g*̀☆λH∗UbΩΨ↓■n∆O卐⩾ρq’⇒D♀\u200bχ@̸`Rω1F̆‖θ®♦&c♯Qξ≦·vδε○А∑”μS―≤ΧG✓ǂ–⊕∞➔!ᅟΙ«⁄4©♣E⇑j|υrα»∩£<X$#⇓⩽※̨∘νØΒm◦×⋅¥∖̈f•—¶Φaˆyτ%5ζN─\u2061¢▲ΑøT≈¤∈t∣◆;ɛ↔?i7‐★̌†k-▪Θπ"CzLV(́△∙Γ⋮♂Σs←J∫æ~pʹ±oσe,Λ≧^□+➢d✗▼0Zx‘]>β√↑„M⋯γ−̄ι∅K′l{'
    config['vocab'] = character
    trainer = Trainer(config, pretrained=False)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        
    trainer.train()

if __name__ == '__main__':
    main()
