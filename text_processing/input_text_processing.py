import re
import json
import logging
from pykospacing import Spacing
from kospellpy import spell_init

def word_spacing_correction(text):
    without_spacing_text = text.replace(" ", "")
    spacing = Spacing()
    correct_spacing_text = spacing(without_spacing_text)
    return correct_spacing_text

async def orthography_examination(text):
    spell_checker = spell_init()
    correct_spell_text = spell_checker(text)
    return correct_spell_text
