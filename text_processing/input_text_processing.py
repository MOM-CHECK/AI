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

async def orthography_examination(text):
    spell_checker = spell_init()
    correct_spell_text = spell_checker(text)
    return correct_spell_text

def morpheme_analyzer(text):
    okt = Okt()
    morphemes = okt.morphs(text, stem=True)
    morpheme_text = ' '.join(morphemes)
    morpheme_text = re.sub(r'\s+\.', '.', morpheme_text)
    return morpheme_text