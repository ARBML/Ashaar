from tnkeeh import clean_data


class Poet:
    name = None
    bio = None
    poems = None
    age = None
    url = None
    location = None


class Poem:
    baits = None
    title = None
    theme = None
    bahr = None
    url = None

    def _clean(self, item):
        pass

    def __eq__(self, other):
        if not isinstance(other, Poem):
            return False

        return self.title == other.title
