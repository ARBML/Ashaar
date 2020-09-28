import re

from farasa.segmenter import FarasaSegmenter


class AshaarPreprocessor:
    def __init__(self):
        pass
        # self.segmenter = FarasaSegmenter(interactive=True, logging_level="error")

    def _remove_diacritics(self, text):
        text = re.sub(r"[ًٌٍَََُِّْ]", "", text)
        return text

    def _remove_special_chars(self, text, execluded_chars=[]):
        return re.compile(
            "([^\n\u0621-\u064A0-9 " + ("").join(execluded_chars) + "])"
        ).sub("", text)

    def _segment(self, text):
        return self.segmenter.segment(text).replace("+", " ")

    def _normalize(self, text):
        normalized_text = re.sub("[إأٱآا]", "ا", text)
        normalized_text = re.sub("ـ", "", text)
        normalized_text = re.sub("ى", "ي", normalized_text)
        normalized_text = re.sub("ؤ", "ء", normalized_text)
        normalized_text = re.sub("ئ", "ء", normalized_text)
        normalized_text = re.sub("ه", "ة", normalized_text)
        return normalized_text

    def preprocess(self, text):
        """
        This method will do the following:
        - remove diacritics
        - remove special chars
        - segment text using farasa
        - normalize
        """
        preprocessed_text = text.strip()
        preprocessed_text = self._remove_diacritics(preprocessed_text)
        preprocessed_text = self._remove_special_chars(preprocessed_text)
        # preprocessed_text = self._segment(preprocessed_text)
        preprocessed_text = self._normalize(preprocessed_text)
        return preprocessed_text
