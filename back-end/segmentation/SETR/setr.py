# https://github.com/gupta-abhay/setr-pytorch
import torch
import torch.nn as nn

from backbone import FixedPositionalEncoding, LearnedPositionalEncoding, IntermediateSequential, TransformerModel


__all__ = [
    'SETR_Naive_S',
    'SETR_Naive_L',
    'SETR_Naive_H',
    'SETR_PUP_S',
    'SETR_PUP_L',
    'SETR_PUP_H',
    'SETR_MLA_S',
    'SETR_MLA_L',
    'SETR_MLA_H',
]


class SegmentationTransformer(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=False,
        # positional_encoding_type="learned",
        positional_encoding_type="fixed",
    ):
        super(SegmentationTransformer, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels

        self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        if self.conv_patch_representation:
            self.conv_x = nn.Conv2d(
                self.num_channels,
                self.embedding_dim,
                kernel_size=(self.patch_dim, self.patch_dim),
                stride=(self.patch_dim, self.patch_dim),
                padding=self._get_padding(
                    'VALID', (self.patch_dim, self.patch_dim),
                ),
            )
        else:
            self.conv_x = None

    def _init_decode(self):
        raise NotImplementedError("Should be implemented in child class!!")

    def encode(self, x):
        n, c, h, w = x.shape
        if self.conv_patch_representation:
            # combine embedding w/ conv patch distribution
            x = self.conv_x(x)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.embedding_dim)
        else:
            x = (
                x.unfold(2, self.patch_dim, self.patch_dim)
                .unfold(3, self.patch_dim, self.patch_dim)
                .contiguous()
            )
            x = x.view(n, c, -1, self.patch_dim ** 2)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.flatten_dim)
            x = self.linear_encoding(x)

        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)

        return x, intmd_x

    def decode(self, x):
        raise NotImplementedError("Should be implemented in child class!!")

    def forward(self, x, auxillary_output_layers=None):
        encoder_output, intmd_encoder_outputs = self.encode(x)
        decoder_output = self.decode(
            encoder_output, intmd_encoder_outputs, auxillary_output_layers
        )

        if auxillary_output_layers is not None:
            auxillary_outputs = {}
            for i in auxillary_output_layers:
                val = str(2 * i - 1)
                _key = 'Z' + str(i)
                auxillary_outputs[_key] = intmd_encoder_outputs[val]

            return decoder_output, auxillary_outputs

        return decoder_output

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class SETR_Naive(SegmentationTransformer):
    def __init__(
        self,
        img_dim=15,
        patch_dim=1,
        num_channels=9,
        num_classes=1,
        embedding_dim=256,
        num_heads=16,
        num_layers=6,
        hidden_dim=512,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=False,
        # positional_encoding_type="learned",
        positional_encoding_type="fixed",
    ):
        super(SETR_Naive, self).__init__(
            img_dim=img_dim,
            patch_dim=patch_dim,
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            conv_patch_representation=conv_patch_representation,
            positional_encoding_type=positional_encoding_type,
        )

        self.num_classes = num_classes
        self._init_decode()

    def _init_decode(self):
        self.conv1 = nn.Conv2d(
            in_channels=self.embedding_dim,
            out_channels=self.embedding_dim,
            kernel_size=1,
            stride=1,
            padding=self._get_padding('VALID', (1, 1),),
        )
        self.bn1 = nn.BatchNorm2d(self.embedding_dim)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=self.embedding_dim,
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1,
            padding=self._get_padding('VALID', (1, 1),),
        )
        self.upsample = nn.Upsample(
            scale_factor=self.patch_dim, mode='bilinear'
        )

    def decode(self, x, intmd_x, intmd_layers=None):
        x = self._reshape_output(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x

# if __name__ == "__main__":
#     model = SETR_Naive()
#     model.cuda()
#     model.eval()
#     image = torch.randn(256, 9, 15, 15).cuda()

#     # print(model)
#     print("input:", image.shape)
#     print("output:", model(image).shape)