import torch, random, gc, os, json
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import AdamW
from datetime import timedelta, datetime
from collections import defaultdict
from utils.average_meter import AverageMeter
from utils.functions import formulate_gold_, formulate_gold_ent
from utils.metric import metric_, ent_metric
from scorer.biorelex import evaluate_biorelex
from scorer.ade import evaluate_ade


def get_linear_schedule_with_warmup_two_stage(optimizer, num_warmup_steps_stage_one, num_training_steps_stage_one, num_warmup_steps_stage_two, num_training_steps_stage_two,  last_epoch=-1):
    def lr_lambda(current_step: int):
        current_step += 1
        if current_step < num_training_steps_stage_one:
            if current_step < num_warmup_steps_stage_one:
                return float(current_step) / float(max(1, num_warmup_steps_stage_one))
            return max(
                0.0, float(num_training_steps_stage_one - current_step) / float(max(1, num_training_steps_stage_one - num_warmup_steps_stage_one))
            )
        else:
            current_step = current_step - num_training_steps_stage_one
            if current_step < num_warmup_steps_stage_two:
                return float(current_step) / float(max(1, num_warmup_steps_stage_two))
            return max(
                0.01, float(num_training_steps_stage_two - current_step) / float(max(1, num_training_steps_stage_two - num_warmup_steps_stage_two))
            )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class Trainer(nn.Module):
    def __init__(self, model, data, args, max_epoch, start_eval_epoch):
        super().__init__()
        self.args = args
        self.model = model
        self.data = data

        self.max_epoch = max_epoch
        self.start_eval_epoch = start_eval_epoch
        self.save_model = args.save_model
        os.makedirs(args.checkpoint_directory, exist_ok=True)
        os.makedirs(args.prediction_directory, exist_ok=True)

        self.start_eval = args.start_eval

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        grouped_params = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and 'bert' in n],
                'weight_decay': args.weight_decay,
                'lr': args.encoder_lr,
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and 'bert' in n],
                'weight_decay': 0.0,
                'lr': args.encoder_lr,
            },
            {
                'params': [p for n, p in self.model.named_parameters() if
                           not any(nd in n for nd in no_decay) and 'bert' not in n],
                'weight_decay': args.weight_decay,
                'lr': args.decoder_lr,
            },
            {
                'params': [p for n, p in self.model.named_parameters() if
                           any(nd in n for nd in no_decay) and 'bert' not in n],
                'weight_decay': 0.0,
                'lr': args.decoder_lr,
            }
        ]
        
        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(grouped_params)
        elif args.optimizer == 'AdamW':
            self.optimizer = AdamW(grouped_params)
        else:
            raise Exception("Invalid optimizer.")
        if args.use_gpu:
            self.cuda()

    def train_model(self):
        best_f1 = 0
        train_loader = self.data.train_loader
        train_num = len(train_loader)
        batch_size = self.args.batch_size
        total_batch = train_num // batch_size + 1
        updates_total_stage_one = total_batch * self.args.split_epoch
        updates_total_stage_two = total_batch * (self.max_epoch - self.args.split_epoch)
        scheduler = get_linear_schedule_with_warmup_two_stage(self.optimizer,
                                                    num_warmup_steps_stage_one = self.args.lr_warmup * updates_total_stage_one,
                                                    num_training_steps_stage_one = updates_total_stage_one,
                                                    num_warmup_steps_stage_two = self.args.lr_warmup * updates_total_stage_two,
                                                    num_training_steps_stage_two = updates_total_stage_two)

        start_datetime_str = datetime.now().strftime('%m-%d-%H-%M-%S')
        if self.model.RE:
            print('\n----------- Start RE Training -----------', start_datetime_str)
        else:
            print('\n----------- Start NER Training -----------', start_datetime_str)
        for epoch in range(self.max_epoch):
            # Train
            self.model.train()
            self.model.zero_grad()

            if epoch == 0:
                # print('Freeze Decoder.')
                # for name, param in self.model.decoder.named_parameters():
                #     param.requires_grad = False

                print("Freeze bert weights")
                for name, param in self.model.encoder.model.named_parameters():
                    if "entity" not in name and "triple" not in name:
                        param.requires_grad = False

            if epoch == self.args.split_epoch:
                print("Now, update bert weights.")
                for name, param in self.model.encoder.model.named_parameters():
                    param.requires_grad = True
                if self.args.fix_bert_embeddings:
                    self.model.encoder.model.embeddings.word_embeddings.weight.requires_grad = False
                    self.model.encoder.model.embeddings.position_embeddings.weight.requires_grad = False
                    self.model.encoder.model.embeddings.token_type_embeddings.weight.requires_grad = False

                self.optimizer.__setstate__({'state': defaultdict(dict)})

            print("=== Epoch %d train ===" % epoch, flush=True)
            avg_loss = AverageMeter()
            random.shuffle(train_loader)
            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > train_num:
                    end = train_num
                train_instance = train_loader[start:end]
                # print([ele[0] for ele in train_instance])
                if not train_instance:
                    continue
                input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, targets, _, info = self.model.batchify(train_instance)
                loss = self.model(input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, targets, info, epoch=epoch)[0]
                avg_loss.update(loss.item(), 1)
                # Optimize
                loss.backward()

                if self.args.max_grad_norm != 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                if (batch_id + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
            # for param_group in self.optimizer.param_groups:
            #     print(param_group['lr'])

            print("     Instance: %d; loss: %.4f" % (end, avg_loss.avg), flush=True)
            if epoch >= self.start_eval_epoch:
                # Validation
                print("=== Epoch %d Validation ===" % epoch)
                result = self.eval_model(self.data.valid_loader, self.data.entity_type_alphabet, self.data.relational_alphabet)
                f1 = result['f1'] if self.model.RE and self.args.dataset_name != 'SciERC' else result['entity_f1']
                if f1 >= best_f1:
                    print("Achieving Best Result on Validation Set.", flush=True)
                    if self.save_model:
                        torch.save(self.model.state_dict(), self.args.checkpoint_directory + "/%s.model" % start_datetime_str)
                    best_f1 = f1
                    best_result_epoch = epoch
                    # # Test
                    # print("=== Epoch %d Test ===" % epoch, flush=True)
                    # result = self.eval_model(self.data.test_loader, self.data.entity_type_alphabet, self.data.relational_alphabet)


        end_datetime_str = datetime.now().strftime('%m-%d-%H-%M-%S')
        if self.model.RE:
            print('\n----------- Finish RE Training -----------', end_datetime_str)
        else:
            print('\n----------- Finish NER Training -----------', end_datetime_str)
        # if self.save_model:
        #     torch.save(self.model.state_dict(), self.args.checkpoint_directory + "/%s.model" % end_datetime_str)
        print("Best result on validation set is achieved at epoch %d." % best_result_epoch, flush=True)
        print("=== Final Test === ", flush=True)
        self.load_state_dict(self.args.checkpoint_directory + "/%s.model" % start_datetime_str)
        result = self.eval_model(self.data.test_loader, self.data.entity_type_alphabet, self.data.relational_alphabet,
                                log_fn=os.path.join(self.args.prediction_directory, start_datetime_str), print_pred=self.args.print_pred)
        # self.load_state_dict(self.args.checkpoint_directory + "/%s.model" % end_datetime_str)
        # result = self.eval_model(self.data.test_loader, self.data.entity_type_alphabet, self.data.relational_alphabet)

    def eval_model(self, eval_loader, ent_type_alphabet, relation_alphabet, log_fn=None, print_pred=False):
        if self.args.dataset_name == 'child':
            return self.default_eval_model(eval_loader, ent_type_alphabet, relation_alphabet, log_fn, print_pred, eval_ent=True)
        elif self.args.dataset_name == 'Text2DT':
            return self.default_eval_model(eval_loader, ent_type_alphabet, relation_alphabet, log_fn, print_pred)
        elif self.args.dataset_name == 'biorelex':
            return self.biorelex_eval_model(eval_loader, ent_type_alphabet, relation_alphabet, log_fn, print_pred)
        elif self.args.dataset_name == 'ade':
            return self.ade_eval_model(eval_loader, ent_type_alphabet, relation_alphabet, log_fn, print_pred)
        else:
            return self.default_eval_model(eval_loader, ent_type_alphabet, relation_alphabet, log_fn, print_pred, eval_ent=True)

    def default_eval_model(self, eval_loader, ent_type_alphabet, relation_alphabet, log_fn=None, print_pred=False, eval_ent=False):
        self.model.eval()
        # print(self.model.decoder.query_embed.weight)
        prediction, gold_ = {}, {}
        prediction_ent, gold_ent = {}, {}
        list_text = []
        with torch.no_grad():
            batch_size = self.args.eval_batch_size
            eval_num = len(eval_loader)
            total_batch = eval_num // batch_size + 1
            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > eval_num:
                    end = eval_num
                eval_instance = eval_loader[start:end]
                if not eval_instance:
                    continue
                input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, target, _, info = self.model.batchify(eval_instance, is_test=True)
                # print(target)
                gen_triples, gen_entities = self.model.default_predict(input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, info, ent_type_alphabet)
                # print(batch_id, flush=True)
                if self.model.RE:
                    gold_.update(formulate_gold_(target, info, relation_alphabet))
                    prediction.update(gen_triples)
                if eval_ent:
                    gold_ent.update(formulate_gold_ent(target, info, ent_type_alphabet))
                    prediction_ent.update(gen_entities)
                list_text += info['text']

        scores = dict()
        if eval_ent:
            ent_scores = ent_metric(prediction_ent, gold_ent, list_text, log_fn, print_pred)
            scores.update(ent_scores)
        if self.model.RE:
            rel_scores = metric_(prediction, gold_, list_text, relation_alphabet, log_fn, print_pred)
            scores.update(rel_scores)
        return scores

    def biorelex_eval_model(self, eval_loader, ent_type_alphabet, relation_alphabet, log_fn=None, print_pred=False):
        self.model.eval()

        truth_sentences, pred_sentences, keys = {}, {}, set()
        with torch.no_grad():
            batch_size = self.args.eval_batch_size
            eval_num = len(eval_loader)
            total_batch = eval_num // batch_size + 1
            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > eval_num:
                    end = eval_num
                eval_instance = eval_loader[start:end]
                if not eval_instance:
                    continue
                input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, _, _, info = self.model.batchify(eval_instance, is_test=True)

                ground_truth = dict(zip(info['sent_idx'], info['inst']))
                truth_sentences.update(ground_truth)

                prediction = self.model.biorelex_predict(input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, 
                                                        info, ent_type_alphabet, relation_alphabet)
                pred_sentences.update(prediction)

                keys = keys | set(info['sent_idx'])

        if print_pred:
            predictions = list(pred_sentences.values())
            with open(log_fn, 'w') as f:
                json.dump(predictions, f, indent=True)

        return evaluate_biorelex(truth_sentences, pred_sentences, keys)
    
    def ade_eval_model(self, eval_loader, ent_type_alphabet, relation_alphabet, log_fn=None, print_pred=False):
        self.model.eval()

        truth_sentences, pred_sentences, keys = {}, {}, set()
        with torch.no_grad():
            batch_size = self.args.eval_batch_size
            eval_num = len(eval_loader)
            total_batch = eval_num // batch_size + 1
            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > eval_num:
                    end = eval_num
                eval_instance = eval_loader[start:end]
                if not eval_instance:
                    continue
                input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, _, _, info = self.model.batchify(eval_instance, is_test=True)

                ground_truth = dict(zip(info['sent_idx'], info['inst']))
                truth_sentences.update(ground_truth)

                prediction = self.model.ade_predict(input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, 
                                                        info, ent_type_alphabet, relation_alphabet)
                pred_sentences.update(prediction)

                keys = keys | set(info['sent_idx'])

        if print_pred:
            predictions = list(pred_sentences.values())
            with open(log_fn, 'w') as f:
                json.dump(predictions, f, indent=True)

        return evaluate_ade(truth_sentences, pred_sentences, keys)

    def load_state_dict(self, path):
        self.model.load_state_dict(torch.load(path))

    @staticmethod
    def lr_decay(optimizer, epoch, decay_rate):
        # lr = init_lr * ((1 - decay_rate) ** epoch)
        if epoch != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * (1 - decay_rate)
                # print(param_group['lr'])
        return optimizer
