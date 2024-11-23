from pykospacing import Spacing

def word_spacing_correction(text):
    without_spacing_text = text.replace(" ", "")
    spacing = Spacing()
    correct_spacing_text = spacing(without_spacing_text)
    return correct_spacing_text
