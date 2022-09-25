import config
from sklearn.preprocessing import MultiLabelBinarizer

class Preprocessor(object):
    def __init__(self,datas):
        self.datas = datas

    def pre_process(self):
        sentences, labels_str, labels, indexes = [], [], [], []
        for i in self.datas:
            sentences.append(i.get('sentence'))
            labels.append(i.get('label'))
            indexes.append(i.get('index'))
            labels_str.append(str(i.get('label')))

        encoed_labels = self.label_encoding(labels_str)

        return sentences, encoed_labels, labels, labels_str

    def label_encoding(self, labels_str):
        multi_encoder = MultiLabelBinarizer()
        encoded_labels = multi_encoder.fit_transform(labels_str)
        return encoded_labels