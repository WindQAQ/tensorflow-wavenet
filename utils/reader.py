import numpy as np
import pandas as pd
from . import kaldi_io


class KaldiReader(object):

    @staticmethod
    def read(feat_path):
        feat = {k: v for k, v in kaldi_io.read_mat_scp(feat_path)}
        feat = pd.DataFrame.from_dict(feat, orient='index').reset_index()
        feat.columns = ['utterance', 'feat']
        feat.set_index('utterance', inplace=True)
        num_features = feat.iloc[0]['feat'].shape[1]

        return feat, num_features


class LabelReader(object):

    @staticmethod
    def read(label_path, label_map=None):

        s = set()

        def process(x):
            ret = x.str.split(',')
            s.update(ret.values[0])

            return ret

        label = pd.read_csv(label_path, delimiter=' ', header=None)
        label.columns = ['utterance', 'label']
        label.set_index('utterance', inplace=True)
        label = label.apply(process, axis=1)
        num_labels = len(s)

        if label_map is None:
            label_map = {v: k for k, v in enumerate(sorted(list(s)))}

        label['label'] = label.apply(lambda x: list(
            map(label_map.get, x.values[0])), axis=1)

        return label, num_labels, label_map
