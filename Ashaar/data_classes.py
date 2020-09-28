from collections import Counter

from .preprocessors import AshaarPreprocessor


class Poet:
    def __init__(
        self,
        name=None,
        bio=None,
        poems=None,
        age=None,
        url=None,
        location=None,
    ):
        name = name
        bio = bio
        poems = poems
        age = age
        url = url
        location = location
        self.preprocessor = AshaarPreprocessor()

    def get_poem(self, poem_title):
        preprocessed_title = self.preprocessor.preprocess(poem_title)
        for poem in self.poems:
            if self.preprocessor.preprocess(poem.title) == preprocessed_title:
                return poem
        return None

    def __str__(self):
        return f"{self.name} poet, url: {self.url}"

    def __repr__(self):
        return f"{self.name} poet object, url: {self.url}"

    def __eq__(self, other):
        if not isinstance(other, Poet):
            return False
        "compare by name only!"
        return self.preprocessor.preprocess(self.name) == self.preprocessor.preprocess(
            other.name
        )


class Poem:
    def __init__(
        self,
        baits=None,
        title=None,
        theme=None,
        bahr=None,
        url=None,
    ):
        baits = baits
        title = title
        theme = theme
        bahr = bahr
        url = url
        self.preprocessor = AshaarPreprocessor()

    @property
    def baits_charachter_frequency(self):
        """
        this function returns a dictionary of charachters and their frequency
        """
        preprocessed_baits = list(
            map(
                lambda bait: self.preprocessor.preprocess(bait),
                self.baits,
            )
        )
        counts = Counter()
        for bait in preprocessed_baits:
            bait_chars = bait.replace(" ", "")
            counts.update(bait_chars)

        """sort by values and return"""
        return {
            charachter: frequency
            for charachter, frequency in sorted(
                dict(counts).items(), key=lambda item: item[1]
            )
        }

    @classmethod
    def frequency_baits_abs_difference(cls, first_poem, second_poem):
        first_poem_baits_char_frequency = first_poem.baits_charachter_frequency
        second_poem_baits_char_frequency = second_poem.baits_charachter_frequency
        get_frequency_sum = lambda poem_baits_char_frequency: sum(
            charachter_frequency_tuple[1]
            for charachter_frequency_tuple in poem_baits_char_frequency.items()
        )
        first_poem_baits_char_frequency_sum = get_frequency_sum(
            first_poem_baits_char_frequency
        )
        second_poem_baits_char_frequency_sum = get_frequency_sum(
            second_poem_baits_char_frequency
        )
        return abs(
            first_poem_baits_char_frequency_sum - second_poem_baits_char_frequency_sum
        )

    def __eq__(self, other):
        if not isinstance(other, Poem):
            return False
        "check if titles are the same"
        if self.preprocessor.preprocess(self.title) == self.preprocessor.preprocess(
            other.title
        ):
            return True
        """
        compare baits frequency, if the difference is greater than 50, return false.
        50 is an experimental number. Why we choose 50? We allowed a maximum diff of half bait, the average length of baits is 25 'NOT EXPERIMENTAL' words.
        The average chars per word is 3.5*12.5 ~= 44 ~= 50 with a margin of 6 more words.
        """
        return Poem.frequency_baits_abs_difference(self, other) < 50

    def __str__(self):
        return f"{self.title} poem, url: {self.url}"

    def __repr__(self):
        return f"{self.title} poem object, url: {self.url}"
