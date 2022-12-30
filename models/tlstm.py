import torch
import torch.nn as nn


class TimeLSTM(nn.Module):
    def __init__(self, args):
        super(TimeLSTM, self).__init__()
        self.args = args

        if self.args.modality == "text":
            modality_embedding_size = self.args.TEXT_EMBEDDING_SIZES[
                self.args.text_embeddings_type
            ]

        if self.args.modality == "image":
            modality_embedding_size = self.args.IMAGE_EMBEDDING_SIZES[
                self.args.image_embeddings_type
            ]

        self.modality_projection = nn.Linear(
            modality_embedding_size, self.args.model_args["embedding_size"]
        )

        self.W_all = nn.Linear(
            self.args.model_args["hidden_size"], self.args.model_args["hidden_size"] * 4
        )
        self.U_all = nn.Linear(
            self.args.model_args["embedding_size"],
            self.args.model_args["hidden_size"] * 4,
        )
        self.W_d = nn.Linear(
            self.args.model_args["hidden_size"], self.args.model_args["hidden_size"]
        )

        self.output = nn.Sequential(
            nn.Linear(
                self.args.model_args["embedding_size"],
                self.args.model_args["hidden_size"],
            ),
            nn.GELU(),
            nn.Linear(self.args.model_args["hidden_size"], 1),
        )

    def _g(self, times):
        return 1 / (times + 1.0)

    def forward(self, batch):
        modality_projections = self.modality_projection(batch["modality"])

        b, seq, embed = modality_projections.size()
        h = torch.zeros(
            b, self.args.model_args["hidden_size"], requires_grad=False
        ).cuda()
        c = torch.zeros(
            b, self.args.model_args["hidden_size"], requires_grad=False
        ).cuda()

        outputs = []

        for s in range(seq):
            c_s1 = torch.tanh(self.W_d(c))
            c_s2 = c_s1 * self._g(batch["time"][:, s : s + 1]).expand_as(c_s1)
            c_l = c - c_s1
            c_adj = c_l + c_s2
            outs = self.W_all(h) + self.U_all(modality_projections[:, s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            c_tmp = torch.sigmoid(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * torch.tanh(c)
            outputs.append(h)

        outputs = torch.stack(outputs, 1)
        outputs = outputs[:, -1]
        logits = self.output(outputs)

        return {
            "logits": logits,
            "probas": torch.sigmoid(logits),
        }
