"""Hold all models you wish to train."""
import torch
import torch.nn.functional as F

from torch import nn

from xcpetion import build_xception_backbone


class SimpleNet(nn.Module):
    """Simple Convolutional and Fully Connect network."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(7, 7))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(7, 7))
        self.conv3 = nn.Conv2d(16, 24, kernel_size=(7, 7))
        self.fc1 = nn.Linear(24 * 26 * 26, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, image):
        """Compute a forward pass."""
        first_conv_features = self.pool(F.relu(self.conv1(image)))
        second_conv_features = self.pool(F.relu(self.conv2(
            first_conv_features)))
        third_conv_features = self.pool(F.relu(self.conv3(
            second_conv_features)))
        # flatten all dimensions except batch
        flattened_features = torch.flatten(third_conv_features, 1)
        fully_connected_first_out = F.relu(self.fc1(flattened_features))
        fully_connected_second_out = F.relu(self.fc2(fully_connected_first_out))
        two_way_output = self.fc3(fully_connected_second_out)
        return two_way_output


def get_xception_based_model() -> nn.Module:
    """Return an Xception-Based network.

    (1) Build an Xception pre-trained backbone and hold it as `custom_network`.
    (2) Override `custom_network`'s fc attribute with the binary
    classification head stated in the exercise.
    """
    """INSERT YOUR CODE HERE, overrun return."""
    from collections import OrderedDict

    custom_network = build_xception_backbone(pretrained=True)
    custom_network.requires_grad_(False)
    custom_network.fc = nn.Sequential(OrderedDict([
        ('Linear1', nn.Linear(2048, 1000)),
        ('relu1', nn.ReLU()),
        ('Linear2', nn.Linear(1000, 256)),
        ('relu2', nn.ReLU()),
        ('Linear3', nn.Linear(256, 64)),
        ('relu3', nn.ReLU()),
        ('Linear4', nn.Linear(64, 2)),
        ]))
    return custom_network

if __name__ == "__main__":
    from xcpetion import build_xception_backbone
    from utils import get_nof_params

    model = build_xception_backbone()
    sum_param_backbone = get_nof_params(model)
    print(sum_param_backbone, " - backbone Parameters")

    model_tuned = get_xception_based_model()
    sum_param_tuned = get_nof_params(model_tuned)
    print(sum_param_tuned, " - tuned Parameters")

    print(f"We added {sum_param_tuned-sum_param_backbone} Parameters")