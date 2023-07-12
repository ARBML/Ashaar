import re
from .constants import VALID_ARABIC
from itertools import product, combinations

_whitespace_re = re.compile(r"\s+")


def collapse_whitespace(text):
    text = re.sub(_whitespace_re, " ", text)
    return text


def basic_cleaners(text):
    text = collapse_whitespace(text)
    return text.strip()


# def valid_arabic_cleaners(text):
#     text = filter(lambda char: char in VALID_ARABIC, text)
#     text = collapse_whitespace(''.join(list(text)))
#     return text.strip()

harakat = ["\u0650", "\u064E", "\u064F"]  # [kasra, fatha, damma, ]
sukun = ["\u0652"]  # [sukun]
mostly_saken = [
  "\u0627",
  "\u0648",
  "\u0649",
  "\u064A",
]  # [alef, waw, alef maqsurah, ya'a]

always_saken = [
  "\u0627",
  "\u0649",
]

tnween_chars = [
    "\u064c",
    "\u064d",
    "\u064b",
]  # damm tanween, kasra tanween, fatha tanween, maddah
shadda_chars = ["\u0651"]
all_tashkeel = harakat+tnween_chars+sukun+shadda_chars


all_chars = list("إةابتثجحخدذرزسشصضطظعغفقكلمنهويىأءئؤ ")
prem_chars = harakat + sukun + mostly_saken + tnween_chars + shadda_chars + all_chars

def not_valid_tashkeel_comb(comb):
  all_comb = list(product(harakat+sukun+tnween_chars, repeat = 2))+list(product(shadda_chars+sukun, repeat = 2))
  if comb in all_comb or comb[::-1] in all_comb:
    return True
  else:
    return False

def remove_tanween_on_alef(text):
  text_copy = ""
  for i in range(0, len(text)):

    # if there is shaddah or character followed by alef followed by tanween add
    if i < len(text) - 2 and text[i] in all_chars+shadda_chars and text[i+1] in always_saken and text[i+2] == tnween_chars[2]:
      text_copy += text[i] + tnween_chars[2]
    
    #ignore current harakah if there is alef followed by tanween
    elif i < len(text) - 2 and text[i] in harakat and text[i+1] in always_saken and text[i+2] == tnween_chars[2] : 
      text_copy += tnween_chars[2]

    # if the current char is tanween with alef is the previous character drop tanween
    elif i > 0 and text[i] == tnween_chars[2] and text[i-1] in always_saken:
      continue

    else:
      text_copy += text[i]
  return text_copy

def dont_start_by_harakah(text):
    text_copy = ""
    for i, char in enumerate(text):
      if not(char in all_tashkeel):
        text_copy = text[i:]
        break 
    return text_copy
        
def valid_arabic_cleaners(text):
  prev_text = text
  for i in range(5):
    text = prev_text
    cleaned_text = ""
    text = filter(lambda char: char in VALID_ARABIC, text)
    text = collapse_whitespace(''.join(list(text)))
    text = dont_start_by_harakah(text)
    text = text.strip()
    i = 0
    cnt = 0
    len_text = len(text)
    while( i < len_text):
      if text[i] in all_tashkeel:
        cnt += 1 
      else:
        cnt = 0

      # don't allow three consecutive tashkeel
      if cnt > 2:
        i+= 1
        continue

      # remove second tanween and sukun
      if i > 1 and text[i] in tnween_chars+sukun and  text[i-2] in tnween_chars+sukun:
        i += 1
        continue 
      
      # don't allow harakah followed by shaddah or tanween
      if i < len(text) - 1 and text[i] in harakat and  text[i+1] in tnween_chars+sukun+shadda_chars:
        i += 1
        continue
      
      # don't allow harkah on space
      if i> 0 and text[i] in all_tashkeel and text[i-1] == " " :
        i += 1
        continue

      # only allow permissable combinations
      if not_valid_tashkeel_comb((text[i], text[i-1])):
        i+=1
        continue

      # don't allow harkah on alef, alef maqsura, if there is no tashkeel before move it back
      if i> 1 and text[i] in harakat and text[i-1] in always_saken :
        if text[i-2] in all_tashkeel: # in case there is a tashkeelah before alef
          continue
        else:
          cleaned_text = text[:i-1]+text[i]+ always_saken[always_saken.index(text[i-1])]
          i += 1 
         
      if i < len(text):
        cleaned_text+= text[i]
        i += 1
    
    # only allow tanween before alef
    cleaned_text = remove_tanween_on_alef(cleaned_text)
    cleaned_text = re.sub(r" +", " ", cleaned_text).strip()
    if prev_text == cleaned_text:
      break
    else:
      prev_text = cleaned_text 
  return cleaned_text