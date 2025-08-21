import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from transformers import Wav2Vec2Model
from pooling import MultiHeadAttentionPooling
from mamba_ssm import Mamba
from multiconv_cgmlp import MultiConvolutionalGatingMLP

class SSLClassifier(pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.feature_extractor = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
        self.cgmlp = MultiConvolutionalGatingMLP(size=128,
                                                 linear_units=1024,
                                                 arch_type="concat_fusion",
                                                 kernel_sizes="7,15,23,31",
                                                 merge_conv_kernel=31,
                                                 use_non_linear=True,
                                                 dropout_rate=0.1,
                                                 use_linear_after_conv=True,
                                                 activation="silu",
                                                 gate_activation="silu"
                                                 )
        self.feature_projection = nn.Linear(1024, 128)
        self.blocks = nn.ModuleList()
        for _ in range(12):
            self.blocks.append(
                nn.Sequential(
                    Mamba(d_model=128,
                          d_state=16,
                          d_conv=4,
                          expand=2),
                    nn.LayerNorm(128),
                )
        )

        self.last_pooling = MultiHeadAttentionPooling(128)
        self.silu = nn.SiLU(inplace=True)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.SELU(inplace=True),
            nn.Linear(128, 2)
        )
        self.config = config
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]))
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x, output_hidden_states=True)
        hidden_states = torch.stack(x.hidden_states, dim=1)
        hidden_states = self.feature_projection(hidden_states)
        hidden_states = self.silu(hidden_states)
        hidden_states_processed = torch.sum(hidden_states, 1)
        x = self.cgmlp(hidden_states_processed)
        for _, b in enumerate(self.blocks):
            x = x + b(x)
        x = self.last_pooling(x.permute(0, 2, 1))
        x = self.classifier(x.squeeze(-1))
        return x

    def on_epoch_end(self, outputs, phase):
        all_scores = []
        all_labels = []
        all_losses = []
        
        for preds, labels, loss in outputs:
            all_scores.append(preds)
            all_labels.append(labels)
            all_losses.append(loss)
        
        all_scores = torch.cat(all_scores)
        all_labels = torch.cat(all_labels)
        all_losses = torch.stack(all_losses)
        
        all_scores = F.softmax(all_scores, dim=-1)
        self.accuracy(torch.argmax(all_scores, 1), all_labels)
        
        self.log_dict(
            {f"{phase}_loss": all_losses.mean(),
             f"{phase}_accuracy": self.accuracy},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        
    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.training_step_outputs.append((scores, y, loss))

        return loss

    def on_train_epoch_end(self):
        self.on_epoch_end(self.training_step_outputs, phase="train")
        self.training_step_outputs.clear()


    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.validation_step_outputs.append((scores, y, loss))

        return loss
    
    def on_validation_epoch_end(self):
        self.on_epoch_end(self.validation_step_outputs, phase="val")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        self._produce_evaluation_file(batch, batch_idx)

    def _produce_evaluation_file(self, batch, batch_idx):
        x, utt_id = batch
        fname_list = []
        score_list = []
        out = self(x)
        out = F.log_softmax(out, dim=-1)
        ss = out[:, 0]
        bs = out[:, 1]
        llr = bs - ss
        if self.config['evaluation']['task'] == "asvspoof":
            utt_id = tuple(item.split('/')[-1].split('.')[0] for item in utt_id)
        fname_list.extend(utt_id)
        score_list.extend(llr.tolist())
            
        with open(self.config['evaluation']['output_score_file'], "a+") as fh:
            for f, cm in zip(fname_list, score_list):
                fh.write("{} {}\n".format(f, cm))
        fh.close()

    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                self.parameters(), lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
        )

        return optimizer