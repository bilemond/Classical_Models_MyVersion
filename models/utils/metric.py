from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_bleu(translate, reference, references_lens):
    """
    计算翻译句子的的BLEU值
    :param translate: transformer翻译的句子
    :param reference: 标准译文
    :return: BLEU值
    """
    # 定义平滑函数
    translate = translate.tolist()
    reference = reference.tolist()
    smooth = SmoothingFunction()
    references_lens = references_lens.tolist()
    blue_score = []
    for translate_sentence, reference_sentence, references_len in zip(translate, reference, references_lens):
        if 1 in translate_sentence:
            index = translate_sentence.index(1)
        else:
            index = len(translate_sentence)
        blue_score.append(sentence_bleu([reference_sentence[:references_len]], translate_sentence[:index], weights=(0.3, 0.4, 0.3, 0.0), smoothing_function=smooth.method1))
    return blue_score