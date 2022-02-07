import torch
import config


class EntityDataset:
    def __init__(self, texts, pos, tag):
        self.texts = texts
        self.pos = pos
        self.tag = tag

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        pos = self.pos[item]
        tag = self.tag[item]

        ids = []
        target_pos = []
        target_tag = []

        # extending labels to length(sub-word tokens): i.e. Kaushik:ENT-P --> #Ka:ENT-P #us:ENT-P #hik:ENT-P
        for i, s in enumerate(text):
            inputs = config.TOKENIZER.encode(s, add_special_tokens=False)
            len_inputs = len(inputs)
            ids.extend(inputs)
            target_tag.extend([tag[i]]*len_inputs)
            target_pos.extend([pos[i]]*len_inputs)

        # setting max_length of tokens
        ids = ids[:config.MAX_LEN-2]
        target_tag = target_tag[:config.MAX_LEN-2]
        target_pos = target_pos[:config.MAX_LEN-2]

        # special tokens: cls = 101, sep = 102
        ids = [101] + ids + [102]
        target_pos = [0] + target_pos + [0]
        target_tag = [0] + target_tag + [0]

        # initializing mask and token id
        mask = [1]*len(ids)
        token_type_ids = [0]*len(ids)

        # padding everything
        padding_len = config.MAX_LEN - len(ids)
        ids = ids + ([0]*padding_len)
        mask = mask + ([0] * padding_len)
        target_pos = target_pos + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)
        token_type_ids= token_type_ids + ([0] * padding_len)

        return {"ids": torch.tensor(ids, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.long),
                "target_pos": torch.tensor(target_pos, dtype=torch.long),
                "target_tag": torch.tensor(target_tag, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long)
                }











