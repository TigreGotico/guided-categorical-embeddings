import ahocorasick
import joblib


class KeywordFeatures:
    def __init__(self, csv_path=None, ignore_list=None):
        ignore_list = ignore_list or []
        self.ignore_list = ignore_list
        self.bias = {}  # just for logging
        self.automatons = {}
        self._needs_building = []
        self.entities = {}
        if csv_path:
            self.load_entities(csv_path)

    @property
    def labels(self):
        return sorted(list(self.entities.keys()))

    def reset_automatons(self):
        # "untrain" the automatons
        self._needs_building = [name for name in self.automatons]
        self.automatons = {name: ahocorasick.Automaton() for name in self.automatons.keys()}
        for name, samples in self.entities.items():
            for s in samples:
                self.automatons[name].add_word(s.lower(), s)

    def register_entity(self, name, samples):
        """ register runtime entity samples,
            eg from skills"""
        if name not in self.entities:
            self.entities[name] = []
        self.entities[name] += samples
        if name not in self.bias:
            self.bias[name] = []
        self.bias[name] += samples

        if name not in self.automatons:
            self.automatons[name] = ahocorasick.Automaton()
        for s in samples:
            self.automatons[name].add_word(s.lower(), s)

        self._needs_building.append(name)

    def deregister_entity(self, name):
        """ register runtime entity samples,
            eg from skills"""
        if name in self.entities:
            self.entities.pop(name)
        if name in self.bias:
            self.bias.pop(name)
        if name in self.automatons:
            self.automatons.pop(name)
        if name in self._needs_building:
            self._needs_building.remove(name)

    def load_entities(self, csv_path):
        ents = {}
        if isinstance(csv_path, str):
            files = [csv_path]
        else:
            files = csv_path
        data = []
        for csv_path in files:
            with open(csv_path) as f:
                lines = f.read().split("\n")[1:]
                data += [l.split(",", 1) for l in lines if "," in l]

        for n, s in data:
            if n not in ents:
                ents[n] = []
            ents[n].append(s)
            self._needs_building.append(n)

        for k, samples in ents.items():
            self._needs_building.append(k)
            if k not in self.automatons:
                self.automatons[k] = ahocorasick.Automaton()
            for s in samples:
                self.automatons[k].add_word(s.lower(), s)
        self.entities.update(ents)
        return ents

    def match(self, utt):
        for k, automaton in self.automatons.items():
            if k in self._needs_building:
                automaton.make_automaton()

        self._needs_building = []

        utt = utt.lower().strip(".!?,;:")

        for k, automaton in self.automatons.items():
            # skip automatons without registered samples
            if not self.entities.get(k):
                continue

            for idx, v in automaton.iter(utt):
                if len(v) < 3:
                    continue

                if "_name" in k and v.lower() in self.ignore_list:
                    # LOG.debug(f"ignoring {k}:  {v}")
                    continue

                # filter partial words
                if " " not in v:
                    if v.lower() not in utt.split(" "):
                        continue
                if v.lower() + " " in utt or utt.endswith(v.lower()):
                    # if k in self.bias:
                    #    LOG.debug(f"BIAS {k} : {v}")
                    yield k, v

    def extract(self, sentence):
        match = {}
        for k, v in self.match(sentence):
            if k not in match:
                match[k] = v
            elif v in self.bias.get(k, []) or len(v) > len(match[k]):
                match[k] = v
        return match

    def save(self, file_path):
        data = {
            'entities': self.entities,
            'bias': self.bias,
            'automatons': self.automatons,
            '_needs_building': self._needs_building,
            'ignore_list': self.ignore_list
        }
        joblib.dump(data, file_path)

    def load(self, file_path):
        data = joblib.load(file_path)
        self.entities = data['entities']
        self.bias = data['bias']
        self.automatons = data['automatons']
        self._needs_building = data['_needs_building']
        self.ignore_list = data['ignore_list']


if __name__ == "__main__":
    kw = KeywordFeatures()

    # Register some example entities
    kw.register_entity('fruit', ['apple', 'banana', 'cherry'])
    kw.register_entity('color', ['red', 'green', 'blue'])

    # Check output
    print("Labels:", kw.labels)
    print("Match 'I have a red apple':", list(kw.match('I have a red apple')))
    print("Extract 'I have a red apple':", kw.extract('I have a red apple'))

    # Save to file
    kw.save('keyword_features.pkl')

    # Load from file
    kw_loaded = KeywordFeatures()
    kw_loaded.load('keyword_features.pkl')

    # Check output after loading
    print("Labels after loading:", kw_loaded.labels)
    print("Match 'I have a green banana':", list(kw_loaded.match('I have a green banana')))
    print("Extract 'I have a green banana':", kw_loaded.extract('I have a green banana'))
