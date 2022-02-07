import config
import torch
import transformers
from dataset import EntityDataset

def loss_fn(output, target, mask, num_labels):
    criteria = torch.nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(active_loss,
                                target.view(-1),
                                torch.tensor(criteria.ignore_index).type_as(target))
    loss = criteria(active_logits, active_labels)
    return loss
    

class NERModel(torch.nn.Module):
    def __init__(self, num_tag, num_pos):
        super(NERModel, self).__init__()
        self.num_tag = num_tag
        self.num_pos = num_pos
        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL_PATH, return_dict=False)
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.2)
        self.out_tag = torch.nn.Linear(768, self.num_tag)
        self.out_pos = torch.nn.Linear(768, self.num_pos)

    def forward(self, ids, mask, token_type_ids, target_tag, target_pos):
        output_seq, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        drop_pos = self.drop1(output_seq)
        drop_tag = self.drop2(output_seq)

        # final layer output
        pred_pos = self.out_pos(drop_pos)
        pred_tag = self.out_tag(drop_tag)

        # calculate loss
        loss_pos = loss_fn(pred_pos, target_pos,  mask, self.num_pos)
        loss_tag = loss_fn(pred_tag, target_tag, mask, self.num_tag)

        fin_loss = (loss_tag+loss_pos)/2
        return pred_pos, pred_tag, fin_loss

    # predict function
    def predict(self, sentences):

        pos = [[0]*len(sent) for sent in sentences]
        tag = [[0]*len(sent) for sent in sentences]
        data = EntityDataset(sentences, pos, sent)
        pred_pos, pred_tag, fin_loss = self.forward(data)
        return pred_pos, pred_tag











