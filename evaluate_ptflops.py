




import torch
import torch.nn as nn
from ptflops import get_model_complexity_info

from option import parser, verify_input_args
from vocab import Vocabulary
from evaluate import load_model

args = verify_input_args(parser.parse_args())


args, model, vocab = load_model(args)


inputs_size = ((3,224,224), (3,224,224), 20)
def input_constructor(inputs_res):
    src_res, trg_res, length = inputs_res
    images_src = torch.ones(()).new_empty((1, *src_res))
    images_trg = torch.ones(()).new_empty((1, *trg_res))
    sentences  = torch.ones((1, length)).long()
    lengths    = torch.tensor([length])

    return {'images_src':images_src, 'images_trg':images_trg, 'sentences':sentences, 'lengths':lengths}


for para in model.parameters(): para.requires_grad = True


freeze_list = []
if args.model_version not in ['EM-only', 'ARTEMIS', 'TIRG']:
    freeze_list += ['Transform_m', 'Attention_EM']
if args.model_version not in ['IS-only', 'ARTEMIS', 'TIRG']:
    freeze_list += ['Attention_IS']
for module in freeze_list:
    for para in getattr(model, module).parameters(): para.requires_grad = False


ignore_list = [nn.GRU]


macs, params = get_model_complexity_info(model, inputs_size, as_strings=True,
        input_constructor=input_constructor, ignore_modules=ignore_list,
        print_per_layer_stat=False, verbose=True)

print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))