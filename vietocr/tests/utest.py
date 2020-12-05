from vietocr.loader.dataloader_v1 import DataGen
from vietocr.model.vocab import Vocab

def test_loader():
    with open("table_ocr/dict.txt", "r") as f:
        t = []
        for i in f.readlines():
            t.append(i.strip('\n'))
        character = set(t)
        character.update('\u2028')
    
    vocab = Vocab(chars=character)
    s_gen = DataGen('./vietocr/tests/', 'sample.txt', vocab, 'cuda:0', 32, 512)

    iterator = s_gen.gen(30)
    for batch in iterator:
        assert batch['img'].shape[1]==3, 'image must have 3 channels'
        assert batch['img'].shape[2]==32, 'the height must be 32'
        print(batch['img'].shape, batch['tgt_input'].shape, batch['tgt_output'].shape, batch['tgt_padding_mask'].shape)

if __name__ == '__main__':
    test_loader()
