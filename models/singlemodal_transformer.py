import torch
import torch.nn as nn
import nomenclature
from models.layers.attention import BertCrossattLayer, LXRTXLayer
from models.time2vec import Time2Vec


class SingleModalTransformer(torch.nn.Module):
    def __init__(self, args):
        super(SingleModalTransformer, self).__init__()
        self.args = args

        if self.args.modality == "text":
            modality_embedding_size = self.args.TEXT_EMBEDDING_SIZES[
                self.args.text_embeddings_type
            ]

        if self.args.modality == "image":
            modality_embedding_size = self.args.IMAGE_EMBEDDING_SIZES[
                self.args.image_embeddings_type
            ]

        if self.args.position_embeddings == "time2vec":
            self.position_embeddings = Time2Vec(args)
        elif self.args.position_embeddings == "learned":
            self.position_embeddings = nn.Parameter(
                torch.randn(
                    1,
                    self.args.window_size,
                    self.args.final_encoder_args["embedding_size"],
                )
            )

        self.modality_projection = torch.nn.Linear(
            modality_embedding_size, self.args.final_encoder_args["embedding_size"]
        )

        self.final_transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                self.args.final_encoder_args["embedding_size"],
                self.args.final_encoder_args["n_heads"],
                activation="gelu",
                batch_first=True,
                norm_first=True,
                dropout=self.args.final_encoder_args["dropout_prob"],
                dim_feedforward=4 * self.args.final_encoder_args["embedding_size"],
            ),
            self.args.final_encoder_args["n_layers"],
        )

        self.output_classification = torch.nn.Linear(
            self.args.final_encoder_args["embedding_size"], 1
        )

    def _g(self, times):
        return 1 / (times + 1.0)

    def forward(self, batch):
        modality_feats = self.modality_projection(batch["modality"])

        if self.args.position_embeddings == "time2vec":
            position_embeddings = self.position_embeddings[batch["time"]]
        elif self.args.position_embeddings == "learned":
            position_embeddings = self.position_embeddings
        else:
            position_embeddings = torch.zeros_like(modality_feats)

        modality_feats = modality_feats + position_embeddings

        final_vector = self.final_transformer(modality_feats)
        final_vector = final_vector.mean(dim=1)

        output = self.output_classification(final_vector)
        output_proba = torch.sigmoid(output)

        return {"logits": output, "probas": output_proba}
