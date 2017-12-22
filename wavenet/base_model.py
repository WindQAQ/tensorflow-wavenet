import tensorflow as tf


class BaseModel(object):
    """Base model.
    """

    def __init__(self, num_features, num_classes):
        self.num_features = num_features
        self.num_classes = num_classes


class CTCModel(BaseModel):
    """Tensorflow model with connectionist temporal classification (CTC) loss.
    """

    def __init__(self, num_features, num_classes):
        super(CTCModel, self).__init__(num_features, num_classes)
        self._build_graph()

    def _build_graph(self):
        """Build the computational gragh.
        """

        self._creat_placeholders()
        logits = self._create_logits()
        self._create_ctc_loss(logits)
        self._create_optimizer()
        self._create_eval(logits)

    def _creat_placeholders(self):
        """Create placeholders including inputs, labels and sequence length.
        """

        with tf.variable_scope('placeholder'):
            self._inputs = tf.placeholder(
                tf.float32, shape=[None, None, self.num_features], name='inputs')

            self._labels = tf.sparse_placeholder(tf.int32, name='labels')

            self._sequence_length = tf.placeholder(
                tf.int32, shape=[None, ], name='sequence_length')

    def _create_logits(self):
        """Create logits according to inputs.
        """

        raise NotImplementedError(
            'This is the abstract method. Subclasses should implement this.')

    def _create_ctc_loss(self, logits):
        """Create CTC loss between labels and logits.
        """

        with tf.variable_scope('loss'):
            self._loss = tf.reduce_mean(tf.nn.ctc_loss(
                self.labels, logits, self.sequence_length))

    def _create_optimizer(self):
        """Create adam optimizer.
        """

        with tf.variable_scope('optimizer'):
            self._optimizer = tf.train.AdamOptimizer(name='optimizer')

    def _create_eval(self, logits):
        """Create evaluation terms including decoded outputs and 
           edit distance between decoded outputs and labels.
        """

        with tf.variable_scope('eval'):
            self.decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(
                logits, self.sequence_length)

            self.edit_distance = tf.reduce_mean(tf.edit_distance(
                tf.cast(self.decoded[0], tf.int32), self.labels))

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def sequence_length(self):
        return self._sequence_length

    @property
    def loss(self):
        return self._loss

    @property
    def optimizer(self):
        return self._optimizer
