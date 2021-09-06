import six

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(char)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0xFF00 and cp <= 0xFFEF) or  # added by daidai
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True
    return False


def is_alpha_char(char):
    if 'a' <= char <= 'z':
        return True
    if 'A' <= char <= 'Z':
        return True
    return False


def basic_tokenize(text):
    tokens = []
    lst_char = ''
    for char in text:
        if is_chinese_char(char):
            tokens.append(char)
        elif char.isdigit():
            if lst_char.isdigit():
                tokens[-1] += char
            else:
                tokens.append(char)
        elif is_alpha_char(char):
            if is_alpha_char(lst_char):
                tokens[-1] += char
            else:
                tokens.append(char)
        else:
            tokens.append(char)
        lst_char = char
    return tokens
