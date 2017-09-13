class Y(object):
    def __init__(self, values, filter_size=5):
        self.values = values.copy()
        self.values = use_thresholds(filters.median_filter(self.values, filter_size))
        self.segments = get_segments(self.values)
        self.seg_len = [len(s) for s in self.segments]
        self.increases = get_increases(self.segments, 1)
        self.inc_len = [len(i) for i in self.increases]
        self.decreases = get_decreases(self.segments)
        self.dec_len = [len(i) for i in self.decreases]
        self.len_extractors = [
            len, 
            np.average, 
            np.min, 
            np.max,
            np.sum,
        ]
        self.len_extractors = [lambda x: e(x) / 300 for e in self.len_extractors]
        
    def _get_features_from_collection(self, collection, extractors=None):
        if extractors is None:
            extractors = self.len_extractors
        if not len(collection):
            return [0] * len(extractors)
        return [e(collection) for e in extractors]
        
    def _get_features_from_values(self):
        return self._get_features_from_collection(self.values, (
            lambda x: np.min(list(x[x > 0]) + [0]), 
            np.max, 
            np.average, 
            lambda x: np.sum(x[x == 1]) / len(x),
        ))
    
    def _get_features_from_seg_len(self):
        return self._get_features_from_collection(self.seg_len)
    
    def _get_features_from_inc_len(self):
        return self._get_features_from_collection(self.inc_len)
    
    def _get_features_from_dec_len(self):
        return self._get_features_from_collection(self.dec_len)
    
    def get_features(self):
        return (
            self._get_features_from_values()
            + self._get_features_from_seg_len()
            + self._get_features_from_inc_len()
            + self._get_features_from_dec_len()
        ) + list(self.values[:5]) + list(self.values[-5:])