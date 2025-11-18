import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict, namedtuple
import numpy as np
from torch import Tensor


# ======================
#  Generic MLP Block
# ======================
class FlexibleMLP(nn.Module):
    """A flexible MLP that supports arbitrary hidden dimensions."""

    def __init__(self, input_dim, hidden_layer_sizes):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layer_sizes[:-1]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, hidden_layer_sizes[-1]))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ======================
#  Dual Head Models (Image)
# ======================
class DualHeadImage(nn.Module):
    def __init__(self, num_classes, feat_dims=[16, 8]):
        super().__init__()
        self.resnet = models.resnet18(weights="IMAGENET1K_V1")
        self.resnet.fc = nn.Identity()  # 移除 fc（變成空操作）
        self.encoded_dim = self.resnet.layer4[1].conv2.out_channels  # ResNet layer4 的輸出通道數

        self.feat_head = FlexibleMLP(self.encoded_dim, feat_dims)
        self.clf_head = nn.Linear(self.encoded_dim, num_classes, bias=True)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)  # 這裡是 backbone 的輸出

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        feat_out = self.feat_head(x)  # 額外的特徵提取層
        clf_out = self.clf_head(x)
        return feat_out, clf_out
    
# ======================
#  Dual Head Models (Tabular)
# ======================
class DualHeadTabular(nn.Module):
    def __init__(self, num_classes, input_dim, enc_dims=[64, 32], feat_dims=[16, 8]):
        super().__init__()
        self.encoder = FlexibleMLP(input_dim, enc_dims)
        
        self.encoded_dim = enc_dims[-1]
        self.feat_head = FlexibleMLP(self.encoded_dim, feat_dims)
        self.clf_head = nn.Linear(self.encoded_dim, num_classes, bias=True)

    def forward(self, x):
        encoded = self.encoder(x)
        feat_out = self.feat_head(encoded)
        clf_out = self.clf_head(encoded)
        return feat_out, clf_out
    

# ======================
#  Base Encoder (Tabular)
# ======================
class BaseTabular(nn.Module):
    """Tabular encoder for structured input."""

    def __init__(self, input_dim, enc_dims=[64, 32], feat_dims=[16, 8]):
        super().__init__()
        self.encoder = FlexibleMLP(input_dim, enc_dims)
        self.encoded_dim = enc_dims[-1]

        self.feat_head = FlexibleMLP(self.encoded_dim, feat_dims)

    def forward(self, x):
        return self.encoder(x)

    def extract_features(self, encoded):
        return self.feat_head(encoded)


# ======================
#  Base Encoder (Image)
# ======================
class BaseImage(nn.Module):
    """Image encoder based on ResNet18 backbone."""

    def __init__(self, feat_dims=[16, 8]):
        super().__init__()
        self.resnet = models.resnet18(weights="IMAGENET1K_V1")
        self.resnet.fc = nn.Identity()
        self.encoded_dim = self.resnet.layer4[1].conv2.out_channels

        self.feat_head = FlexibleMLP(self.encoded_dim, feat_dims)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

    def extract_features(self, encoded):
        return self.feat_head(encoded)


# ======================
#  ECOC Transformer
# ======================
class ECOCOutputTransformer(nn.Module):
    """A transformer for the output of the OBD model in order
    to apply the ECOC scheme.

    Parameters
    ----------
    num_classes : int
        Number of classes.
    """

    target_class: Tensor

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        target_class = np.ones((num_classes, num_classes - 1), dtype=np.float32)
        target_class[np.triu_indices(num_classes, 0, num_classes - 1)] = 0.0
        target_class = torch.tensor(target_class, dtype=torch.float32)
        self.register_buffer("target_class", target_class)

    def probas(self, output):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input to the model

        Returns
        -------
        probas : Tensor
            The predicted probability of belonging to each class :math:`P(y = q)`.
        """
        return torch.softmax(self.scores(output), dim=1)

    def scores(self, output):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input to the model

        Returns
        -------
        scores : Tensor
            The negative distance to each class ideal vector, to use
            as class scores.
        """
        return -torch.cdist(output, self.target_class)

    def labels(self, output):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input to the model

        Returns
        -------
        labels : Tensor
            The predicted integer label according to the ECOC assignment
            scheme.
        """
        scores = self.scores(output)
        return scores.argmax(dim=1)



PredictOutput = namedtuple("PredictOutput", ["scores", "probas", "labels"])


# ======================
#  OBDECOC Head
# ======================
class OBDECOCHead(nn.Module):
    """Ordinal Binary Decomposition (OBD) model with ECOC transformation.

    Reference
    ---------
    Barbero et al., "Error-Correcting Output Codes for Ordinal Classification", 2023.
    """

    def __init__(
        self, num_classes: int, base_classifier: nn.Module, base_n_outputs: int
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.base_classifier = base_classifier
        self.obd_output = nn.Sequential(
            OrderedDict(
                [
                    ("penultimate_activation", nn.ReLU()),
                    ("last_linear", nn.Linear(base_n_outputs, num_classes - 1)),
                    ("last_activation", nn.Sigmoid()),
                ]
            )
        )
        self.transformer = ECOCOutputTransformer(num_classes)

    def forward(self, x: Tensor) -> Tensor:
        encoded = self.base_classifier(x)
        feat_output = self.base_classifier.extract_features(encoded)
        logits = self.obd_output(encoded)
        return feat_output, logits

    def predict_from_inputs(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input to the model

        Returns
        -------
        An object with the following attributes.

        scores : Tensor
            The negative distance to each class ideal vector, to use
            as class scores.
        probas : Tensor
            The predicted probability of belonging to each class :math:`P(y = q)`.
        labels : Tensor
            The predicted integer label according to the ECOC assignment
            scheme.
        """
        raw_output = self(x)
        return PredictOutput(
            self.transformer.scores(raw_output),
            self.transformer.probas(raw_output),
            self.transformer.labels(raw_output),
        )

# ======================
#  Example Usage
# ======================
if __name__ == "__main__":
    x_dim = 32
    n_classes = 4
    enc_dims = [64, 32]
    feat_dims = [16, 8]

    base = BaseTabular(input_dim=x_dim, enc_dims=enc_dims, feat_dims=feat_dims)
    model = OBDECOCHead(num_classes=n_classes, base_classifier=base, base_output_dim=enc_dims[-1])

    sample = torch.randn(8, x_dim)
    out = model(sample)
    print({k: v.shape for k, v in out.items()})
