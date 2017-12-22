import tensorflow as tf

from .base_model import CTCModel
from .layers import conv1d
from .layers import dilation_layer


class WaveNet(CTCModel):
    """WaveNet network for speech recognition.
       This model use CTC loss which is helpful for unsegmented data.
    """

    def __init__(self,
                 num_features,
                 num_classes,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 initial_kernel_size=2,
                 kernel_size=2,
                 num_residual_blocks=1,
                 num_dilation_layers=5,
                 use_bias=False):

        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.initial_kernel_size = initial_kernel_size
        self.kernel_size = kernel_size
        self.num_residual_blocks = num_residual_blocks
        self.num_dilation_layers = num_dilation_layers
        self.use_bias = use_bias
        super(WaveNet, self).__init__(num_features, num_classes)

    def _create_logits(self):
        """Create logits based on WaveNet network.
        """

        num_classes = self.num_classes
        residual_channels = self.residual_channels
        dilation_channels = self.dilation_channels
        skip_channels = self.skip_channels
        initial_kernel_size = self.initial_kernel_size
        kernel_size = self.kernel_size
        num_residual_blocks = self.num_residual_blocks
        num_dilation_layers = self.num_dilation_layers
        use_bias = self.use_bias
        skips = []
        outputs = conv1d(self.inputs, residual_channels, initial_kernel_size)

        for block_id in range(num_residual_blocks):
            for i in range(num_dilation_layers):
                dilation_rate = int(2 ** i)
                name = 'residual-block-{}-dilation-rate-{}'.format(
                    block_id, dilation_rate)

                outputs, skip_connections = dilation_layer(outputs,
                                                           residual_channels,
                                                           dilation_channels,
                                                           skip_channels,
                                                           kernel_size,
                                                           dilation_rate,
                                                           use_bias=use_bias,
                                                           name=name)

                skips.append(skip_connections)

        # sum -> relu -> 1x1 conv -> relu -> 1x1 conv -> outputs
        outputs = tf.add_n(skips)
        outputs = tf.nn.relu(outputs)
        outputs = conv1d(outputs, skip_channels, 1,
                         use_bias=use_bias, name='post1-1x1-conv')
        outputs = tf.nn.relu(outputs)
        outputs = conv1d(outputs, num_classes, 1,
                         use_bias=use_bias, name='post2-1x1-conv')

        logits = tf.transpose(outputs, perm=[1, 0, 2])

        return logits