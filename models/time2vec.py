import torch
from torch import nn


class Time2Vec(nn.Module):
    def __init__(self, args):
        super(Time2Vec, self).__init__()
        self.args = args

        self.w0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1, 1))

        self.w1 = nn.parameter.Parameter(
            torch.randn(1, self.args.cross_encoder_args["embedding_size"] - 1)
        )
        self.b1 = nn.parameter.Parameter(
            torch.randn(1, self.args.cross_encoder_args["embedding_size"] - 1)
        )

        if self.args.time2vec_activation == "sin":
            self.f = torch.sin
        elif self.args.time2vec_activation == "cos":
            self.f = torch.cos

    def _g(self, times):
        return 1 / (times + 1)

    def __getitem__(self, tau):
        return self(tau)

    def forward(self, tau):
        tau = self._g(tau)
        v1 = self.f(
            torch.matmul(tau.view(-1, self.args.window_size, 1), self.w1) + self.b1
        )
        v2 = torch.matmul(tau.view(-1, self.args.window_size, 1), self.w0) + self.b0
        concatenated = torch.cat([v1, v2], -1)
        return concatenated
