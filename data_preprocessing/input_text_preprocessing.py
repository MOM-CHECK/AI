import re
import json
import logging
from pykospacing import Spacing
from kospellpy import spell_init
from konlpy.tag import Okt

def word_spacing_correction(text):
    without_spacing_text = text.replace(" ", "")
    spacing = Spacing()
    correct_spacing_text = spacing(without_spacing_text)
    return correct_spacing_text

def orthography_examination(text):
    spell_checker = spell_init()
    try:
        correct_spell_text = spell_checker(text)
        return correct_spell_text
    except Exception as e:
        logging.error("라이브러리 상의 이유로 맞춤법 검사가 어렵습니다.")
        return text

def morpheme_analyzer(text):
    okt = Okt()
    morphemes = okt.morphs(text, stem=True)
    morpheme_text = ' '.join(morphemes)
    morpheme_text = re.sub(r'\s+\.', '.', morpheme_text)
    return morpheme_text

def text_processing(text):
    correct_spell = orthography_examination(text)
    correct_spacing = word_spacing_correction(correct_spell)
    # final_correct_text = morpheme_analyzer(correct_spacing)
    return correct_spacing