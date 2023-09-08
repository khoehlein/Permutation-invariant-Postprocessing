from torch import nn

from model.ensemble.merger import MeanMerger


class EnsembleEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels, bottleneck_channels=None, ensemble_merger=None, dropout=0.2):
        super(EnsembleEncoder, self).__init__()
        if bottleneck_channels is None:
            bottleneck_channels = hidden_channels
        self.member_model = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, (1,)),
            self._activation_layer(hidden_channels),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, (1,)),
            self._activation_layer(hidden_channels),
            nn.Conv1d(hidden_channels, bottleneck_channels, (1,)),
        )
        self.merger = MeanMerger() if ensemble_merger is None else ensemble_merger
        self.ensemble_model = nn.Sequential(
            nn.Linear(self.merger.output_channels(bottleneck_channels), hidden_channels),
            self._activation_layer(hidden_channels),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            self._activation_layer(hidden_channels // 2),
            nn.Linear(hidden_channels // 2, out_channels),
            nn.Softplus()
        )

    def _activation_layer(self, channels):
        return nn.Softplus()

    def forward(self, ensemble):
        features = self.member_model(ensemble)
        features = self.merger(features)
        features = self.ensemble_model(features)
        return features
