import torch
import torch.nn as nn
import nomenclature
from models.layers.attention import BertCrossattLayer, LXRTXLayer
from models.time2vec import Time2Vec


class MultiModalTransformer(torch.nn.Module):
    def __init__(self, args):
        super(MultiModalTransformer, self).__init__()
        self.args = args

        image_embedding_size = self.args.IMAGE_EMBEDDING_SIZES[
            self.args.image_embeddings_type
        ]
        text_embedding_size = self.args.TEXT_EMBEDDING_SIZES[
            self.args.text_embeddings_type
        ]

        if self.args.position_embeddings == "time2vec":
            self.position_embeddings = Time2Vec(args)
        elif self.args.position_embeddings == "learned":
            self.position_embeddings = nn.Parameter(
                torch.randn(
                    1,
                    self.args.window_size,
                    self.args.cross_encoder_args["embedding_size"],
                )
            )

        self.image_projection = torch.nn.Linear(
            image_embedding_size, self.args.cross_encoder_args["embedding_size"]
        )
        self.text_projection = torch.nn.Linear(
            text_embedding_size, self.args.cross_encoder_args["embedding_size"]
        )

        # this is cross_encoder
        self.layers = torch.nn.ModuleList(
            [LXRTXLayer(args) for _ in range(self.args.cross_encoder_args["n_layers"])]
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

    def forward(self, batch):
        extended_image_attention_mask = (1.0 - batch["image_mask"]) * -10000.0
        extended_text_attention_mask = (1.0 - batch["text_mask"]) * -10000.0

        all_lang_feats = self.text_projection(batch["text_embeddings"])
        all_visn_feats = self.image_projection(batch["image_embeddings"])

        if self.args.position_embeddings == "time2vec":
            position_embeddings = self.position_embeddings[batch["time"]]
        elif self.args.position_embeddings == "learned":
            position_embeddings = self.position_embeddings
        else:
            position_embeddings = torch.zeros_like(all_lang_feats)

        lang_feats = all_lang_feats + position_embeddings
        visn_feats = all_visn_feats + position_embeddings

        for layer_module in self.layers:
            lang_feats, visn_feats = layer_module(
                lang_feats,
                extended_text_attention_mask,
                visn_feats,
                extended_image_attention_mask,
            )

        cross_modal_vectors = lang_feats

        final_vector = self.final_transformer(cross_modal_vectors)
        final_vector = final_vector.mean(dim=1)

        output = self.output_classification(final_vector)
        output_proba = torch.sigmoid(output)

        return {"logits": output, "probas": output_proba}
