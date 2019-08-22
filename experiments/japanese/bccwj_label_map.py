from data_utils.vocab import Vocabulary

NERLabelMapper = Vocabulary(True)
NERLabelMapper.add('X')
NERLabelMapper.add('[CLS]')
NERLabelMapper.add('[SEP]')
NERLabelMapper.add('O')
with open('experiments/japanese/ne_labels.txt') as f:
    netypes = [l for l in f.read().split('\n') if l.strip()]
    for netype in netypes:
        NERLabelMapper.add(netype)

NERALLLabelMapper = Vocabulary(True)
NERALLLabelMapper.add('X')
NERALLLabelMapper.add('[CLS]')
NERALLLabelMapper.add('[SEP]')
NERALLLabelMapper.add('O')
with open('experiments/japanese/ene_labels_all.txt') as f:
    enetypes = [l for l in f.read().split('\n') if l.strip()]
    for enetype in enetypes:
        NERALLLabelMapper.add(enetype)

ChunkingLabelMapper = Vocabulary(True)
ChunkingLabelMapper.add("X")
ChunkingLabelMapper.add("[CLS]")
ChunkingLabelMapper.add("[SEP]")
ChunkingLabelMapper.add("O")
ChunkingLabelMapper.add("B-NP")
ChunkingLabelMapper.add("I-NP")

POSLabelMapper = Vocabulary(True)
POSLabelMapper.add("X")
POSLabelMapper.add("[CLS]")
POSLabelMapper.add("[SEP]")
POSLabelMapper.add("O")
POSLabelMapper.add("(null)")
POSLabelMapper.add("判定詞")
POSLabelMapper.add("副詞")
POSLabelMapper.add("助動詞")
POSLabelMapper.add("助詞")
POSLabelMapper.add("動詞")
POSLabelMapper.add("名詞")
POSLabelMapper.add("形容詞")
POSLabelMapper.add("感動詞")
POSLabelMapper.add("指示詞")
POSLabelMapper.add("接尾辞")
POSLabelMapper.add("接続詞")
POSLabelMapper.add("接頭辞")
POSLabelMapper.add("特殊")
POSLabelMapper.add("連体詞")

FinePOSLabelMapper = Vocabulary(True)
FinePOSLabelMapper.add("X")
FinePOSLabelMapper.add("[CLS]")
FinePOSLabelMapper.add("[SEP]")
FinePOSLabelMapper.add("O")
FinePOSLabelMapper.add("*")
FinePOSLabelMapper.add("イ形容詞接頭辞")
FinePOSLabelMapper.add("サ変名詞")
FinePOSLabelMapper.add("ナ形容詞接頭辞")
FinePOSLabelMapper.add("人名")
FinePOSLabelMapper.add("副助詞")
FinePOSLabelMapper.add("副詞形態指示詞")
FinePOSLabelMapper.add("副詞的名詞")
FinePOSLabelMapper.add("動詞性接尾辞")
FinePOSLabelMapper.add("動詞接頭辞")
FinePOSLabelMapper.add("句点")
FinePOSLabelMapper.add("名詞形態指示詞")
FinePOSLabelMapper.add("名詞性名詞助数辞")
FinePOSLabelMapper.add("名詞性名詞接尾辞")
FinePOSLabelMapper.add("名詞性特殊接尾辞")
FinePOSLabelMapper.add("名詞性述語接尾辞")
FinePOSLabelMapper.add("名詞接頭辞")
FinePOSLabelMapper.add("固有名詞")
FinePOSLabelMapper.add("地名")
FinePOSLabelMapper.add("形容詞性名詞接尾辞")
FinePOSLabelMapper.add("形容詞性述語接尾辞")
FinePOSLabelMapper.add("形式名詞")
FinePOSLabelMapper.add("括弧始")
FinePOSLabelMapper.add("括弧終")
FinePOSLabelMapper.add("接続助詞")
FinePOSLabelMapper.add("数詞")
FinePOSLabelMapper.add("時相名詞")
FinePOSLabelMapper.add("普通名詞")
FinePOSLabelMapper.add("格助詞")
FinePOSLabelMapper.add("空白")
FinePOSLabelMapper.add("終助詞")
FinePOSLabelMapper.add("組織名")
FinePOSLabelMapper.add("記号")
FinePOSLabelMapper.add("読点")
FinePOSLabelMapper.add("連体詞形態指示詞")

GLOBAL_MAP = {
    'ner': NERLabelMapper,
    'nerall': NERALLLabelMapper,
    'chunking': ChunkingLabelMapper,
    'pos': POSLabelMapper,
    'finepos': FinePOSLabelMapper,
}

METRIC_META = {
    'ner': [7, 8, 9, 10, 11, 12],
    'nerall': [7, 8, 9, 10, 11, 12],
    'chunking': [7, 8, 9, 10, 11, 12],
    'pos': [7, 8, 9, 10, 11, 12],
    'finepos': [7, 8, 9, 10, 11, 12],
}

SAN_META = {
    'ner': 2,
    'nerall': 2,
    'chunking': 2,
    'pos': 2,
    'finepos': 2,
}