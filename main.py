import argparse, os

def str2bool(v):
    return v.lower() in ('true')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    return arg

def get_args():
    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    return args, unparsed

parser = argparse.ArgumentParser()
data_arg = add_argument_group('Data')

data_arg.add_argument('--dataset_name', type=str, default="Text2DT")
data_arg.add_argument('--train_file', type=str, default="./data/Text2DT/train_dev.txt")
data_arg.add_argument('--valid_file', type=str, default="./data/Text2DT/test.txt")
data_arg.add_argument('--test_file', type=str, default="./data/Text2DT/test.txt")

data_arg.add_argument('--generated_data_directory', type=str, default="./data/generated_data/")
data_arg.add_argument('--checkpoint_directory', type=str, default="./checkpoints")
data_arg.add_argument('--prediction_directory', type=str, default="./predictions")
data_arg.add_argument('--bert_directory', type=str, default="/disk3/hyx/huggingface/bioLinkBert_base_uncase")
data_arg.add_argument("--partial", type=str2bool, default=False)
learn_arg = add_argument_group('Learning')
learn_arg.add_argument('--model_name', type=str, default="Set-Prediction-Networks")
learn_arg.add_argument('--repeat_gt_entities', type=int, default=25)
learn_arg.add_argument('--repeat_gt_triples', type=int, default=15)
learn_arg.add_argument('--num_generated_triples', type=int, default=10)
learn_arg.add_argument('--entity_queries_num', type=int, default=30)
learn_arg.add_argument('--num_decoder_layers', type=int, default=3)
learn_arg.add_argument('--matcher', type=str, default="avg", choices=['avg', 'min'])
learn_arg.add_argument('--hybrid', type=str2bool, default=True)
learn_arg.add_argument('--consistency_loss_weight', type=float, default=1)
learn_arg.add_argument('--start_consistency_epoch', type=int, default=0)
learn_arg.add_argument('--na_rel_coef', type=float, default=1)
learn_arg.add_argument('--rel_loss_weight', type=float, default=1)
learn_arg.add_argument('--head_ent_loss_weight', type=float, default=0.5)
learn_arg.add_argument('--tail_ent_loss_weight', type=float, default=0.5)
learn_arg.add_argument('--na_ent_coef', type=float, default=1)
learn_arg.add_argument('--ent_type_loss_weight', type=float, default=1)
learn_arg.add_argument('--ent_span_loss_weight', type=float, default=1)
learn_arg.add_argument('--ent_part_loss_weight', type=float, default=1)
learn_arg.add_argument('--head_part_loss_weight', type=float, default=1)
learn_arg.add_argument('--tail_part_loss_weight', type=float, default=1)
learn_arg.add_argument('--head_tail_type_loss_weight', type=float, default=1)
learn_arg.add_argument('--head_tail_entID_loss_weight', type=float, default=1)
learn_arg.add_argument('--relID_loss_weight', type=float, default=1)
learn_arg.add_argument('--ent_have_rel_loss_weight', type=float, default=1)
learn_arg.add_argument('--start_ent_have_rel_epoch', type=int, default=0)
learn_arg.add_argument('--stop_ent_have_rel_epoch', type=int, default=100)
learn_arg.add_argument('--fix_bert_embeddings', type=str2bool, default=True)
learn_arg.add_argument('--batch_size', type=int, default=8)
learn_arg.add_argument('--eval_batch_size', type=int, default=8)
learn_arg.add_argument('--max_epoch', type=int, default=50)
learn_arg.add_argument('--start_eval', type=int, default=40)
learn_arg.add_argument('--gradient_accumulation_steps', type=int, default=1)
learn_arg.add_argument('--decoder_lr', type=float, default=2e-5)
learn_arg.add_argument('--encoder_lr', type=float, default=1e-5)
learn_arg.add_argument('--weight_decay', type=float, default=1e-5)
learn_arg.add_argument('--dropout', type=float, default=0.3)
learn_arg.add_argument('--max_grad_norm', type=float, default=0)
learn_arg.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'])
learn_arg.add_argument('--save_model', type=str2bool, default=True)
learn_arg.add_argument('--lr_warmup', type=float, default=0.1,
                        help="Proportion of total train iterations to warmup in linear increase/decrease schedule")
# PIQN
learn_arg.add_argument('--prop_drop', type=float, default=0.1, help="Probability of dropout used in piqn")
learn_arg.add_argument('--freeze_transformer', type=str2bool, default=False, help="Freeze BERT weights")
learn_arg.add_argument('--pos_size', type=int, default=25)
learn_arg.add_argument('--char_lstm_layers', type=int, default=1)
learn_arg.add_argument('--lstm_layers', type=int, default=2)
learn_arg.add_argument('--char_size', type=int, default=25)
learn_arg.add_argument('--char_lstm_drop', type=float, default=0.2)
learn_arg.add_argument('--use_glove', type=str2bool, default=False)
learn_arg.add_argument('--use_pos', type=str2bool, default=False)
learn_arg.add_argument('--use_char_lstm', type=str2bool, default=False)
learn_arg.add_argument('--use_lstm', type=str2bool, default=False)
learn_arg.add_argument('--pool_type', type=str, default = "max")
learn_arg.add_argument('--wordvec_path', type=str, default = "../glove/glove.6B.300d.txt")
learn_arg.add_argument('--share_query_pos', type=str2bool, default=True)
learn_arg.add_argument('--use_token_level_encoder', type=str2bool, default=True)
learn_arg.add_argument('--num_token_ent_rel_layer', type=int, default=2)
learn_arg.add_argument('--num_token_ent_layer', type=int, default=2)
learn_arg.add_argument('--num_token_rel_layer', type=int, default=1)
learn_arg.add_argument('--num_token_head_tail_layer', type=int, default=2)
learn_arg.add_argument('--use_entity_attention', type=str2bool, default=True)
learn_arg.add_argument('--use_aux_loss', type=str2bool, default=True)
learn_arg.add_argument('--split_epoch', type=int, default=0, help="")
# PIQN EntityAwareConfig
learn_arg.add_argument('--mask_ent2tok', type=str2bool, default=True)
learn_arg.add_argument('--mask_tok2ent', type=str2bool, default=False)
learn_arg.add_argument('--mask_ent2ent', type=str2bool, default=True)
learn_arg.add_argument('--mask_entself', type=str2bool, default=True)
learn_arg.add_argument('--word_mask_ent2tok', type=str2bool, default=True)
learn_arg.add_argument('--word_mask_tok2ent', type=str2bool, default=False)
learn_arg.add_argument('--word_mask_ent2ent', type=str2bool, default=True)
learn_arg.add_argument('--word_mask_entself', type=str2bool, default=True)
learn_arg.add_argument('--entity_aware_attention', type=str2bool, default=False)
learn_arg.add_argument('--entity_aware_selfout', type=str2bool, default=False)
learn_arg.add_argument('--entity_aware_intermediate', type=str2bool, default=False)
learn_arg.add_argument('--entity_aware_output', type=str2bool, default=False)
learn_arg.add_argument('--use_entity_pos', type=str2bool, default=True)
learn_arg.add_argument('--use_entity_common_embedding', type=str2bool, default=True)

evaluation_arg = add_argument_group('Evaluation')
evaluation_arg.add_argument('--n_best_size', type=int, default=50)
evaluation_arg.add_argument('--max_span_length', type=int, default=12)
evaluation_arg.add_argument('--ent_class_ths', type=float, default=0)
evaluation_arg.add_argument('--rel_class_ths', type=float, default=0)
evaluation_arg.add_argument('--boundary_ths', type=float, default=0)
misc_arg = add_argument_group('MISC')
misc_arg.add_argument('--refresh', type=str2bool, default=True)
misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
misc_arg.add_argument('--print_pred', type=str2bool, default=True)
misc_arg.add_argument('--visible_gpu', type=int, default=1)
misc_arg.add_argument('--random_seed', type=int, default=1)

args, unparsed = get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible_gpu)
for arg in vars(args):
    print(arg, ":",  getattr(args, arg))

import torch
import random
import numpy as np
from utils.data import build_data
from trainer.trainer import Trainer
from models.setpred4RE import SetPred4RE
from models.setpred4RE_cpt import SetPred4RE_cpt


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(args.random_seed)

data = build_data(args)
if 'cpt' in args.bert_directory:
    model = SetPred4RE_cpt(args, data.relational_alphabet.size(), data.entity_type_alphabet.size(), RE=True)
else:
    model = SetPred4RE(args, data.relational_alphabet.size(), data.entity_type_alphabet.size(), RE=True)

RE_trainer = Trainer(model, data, args, args.max_epoch, args.start_eval)

RE_trainer.train_model()

# model_id = '08-24-10-37-29'
# RE_trainer.load_state_dict(args.checkpoint_directory + "/%s.model" % model_id)
# result = RE_trainer.eval_model(data.test_loader, data.entity_type_alphabet, data.relational_alphabet,
#                                 log_fn=os.path.join(args.prediction_directory, model_id), print_pred=args.print_pred)
