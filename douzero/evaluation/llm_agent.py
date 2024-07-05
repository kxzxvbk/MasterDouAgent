import copy
import warnings
import re

from typing import List, Optional

import numpy as np
import torch
from peft import prepare_model_for_int8_training, get_peft_model, TaskType, LoraConfig
from torch import nn
from transformers import LlamaTokenizer, LlamaForCausalLM, StoppingCriteria, StoppingCriteriaList

from douzero.env.env import get_obs
from douzero.evaluation.deep_agent import DeepAgent
from douzero.evaluation.dozero_env.env import GameEnv
from douzero.evaluation.rlcard_agent import RLCardAgent

_RealCard2EnvCard = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                     '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12,
                     'K': 13, 'A': 14, '2': 17, 'B': 20, 'R': 30}

players = {
    'landlord': RLCardAgent('landlord'),
    'landlord_up': DeepAgent('landlord_up', 'baselines/douzero_WP/landlord_up.ckpt'),
    'landlord_down': DeepAgent('landlord_down', 'baselines/douzero_WP/landlord_down.ckpt')
}
SIMUL_ENV = GameEnv(players=players)
INFOSET = None


def is_toolcall_point(input_ids, tokenizer):
    string = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    pattern = r"If I play \[\d*(, \d+)*\],$"
    result = re.findall(pattern, string)
    return len(result) != 0


class StopAtSpecificTokenCriteria(StoppingCriteria):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return is_toolcall_point(input_ids, self.tokenizer)

stopping_criteria = None


def response2txt(resp, is_win):
    if is_win is None:
        post_str = ''
    elif is_win:
        post_str = 'I win.'
    else:
        post_str = 'I lose.'

    if resp[0] is None:
        pre_str = ''
    elif resp[1] is None:
        pre_str = f'the next player will play {resp[0]}.'
    else:
        pre_str = f'the next two players will play {resp[0]} and {resp[1]}.'

    if len(post_str) > 0:
        if len(pre_str) > 0:
            return pre_str + ' ' + post_str
        else:
            return post_str
    else:
        return pre_str


def extract_last_action(sentence: str) -> Optional[str]:
    pattern = r'If I play (\[\d*(?:, \d+)*\]),$'
    result = re.findall(pattern, sentence)
    if len(result) == 0:
        return
    return eval(result[-1])


def call_tool(action) -> str:
    """
    Given the command ``cmd`` and generated sentence, use the tool function to generate the result.
    """
    res = SIMUL_ENV.step_2_farmers(INFOSET, action)
    if res not in INFOSET.legal_actions:
        return "this is not a legal action. "
    if SIMUL_ENV.game_over:
        is_win = len(SIMUL_ENV.info_sets['landlord'].player_hand_cards) == 0
    else:
        is_win = None
    SIMUL_ENV.reset()
    return response2txt(res, is_win)


def tool_generation(model: nn.Module, tokenizer: LlamaTokenizer, prompt: torch.Tensor, max_length: int) -> torch.Tensor:
    """
    Generate a complete sentence given the prompt. This function will automatically call the tools for generation.
    """
    generate_ids = model.generate(inputs=prompt, max_length=max_length,
                                  stopping_criteria=stopping_criteria, do_sample=False)
    while is_toolcall_point(generate_ids, tokenizer):
        current_string = \
            tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # Extract the last command.
        api_command = extract_last_action(current_string)
        if api_command is not None:
            # Call the tool and get the result in string type.
            api_reply = call_tool(api_command)
            # Get the new encoded sentence.
            new_prompt = tokenizer.encode(
                current_string + api_reply,
                max_length=4096,
                truncation=True,
                add_special_tokens=True
            )
            new_prompt = torch.tensor([new_prompt]).cuda()
        else:
            new_prompt = tokenizer.encode(
                current_string,
                max_length=4096,
                truncation=True,
                add_special_tokens=True
            )
            new_prompt = torch.tensor([new_prompt]).cuda()
        # Continue generation.
        generate_ids = model.generate(inputs=new_prompt, max_length=max_length,
                                      stopping_criteria=stopping_criteria, do_sample=False)

    return generate_ids


def tool_generate_txt(model: nn.Module, tokenizer: LlamaTokenizer, prompt: str) -> str:
    """
    Given a prompt, generate the result by using tool functions.
    """
    # Encode the prompt.
    prompt_ids = tokenizer.encode(
        prompt,
        max_length=2048,
        truncation=True,
        add_special_tokens=True
    )
    prompt_ids = torch.tensor([prompt_ids]).cuda()
    # Generate the results.
    generate_ids = tool_generation(model=model, tokenizer=tokenizer, prompt=prompt_ids, max_length=4096)
    total_res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # Remove the prompt part, only return the response.
    return total_res[len(prompt):]


def _char2num(que: List) -> List:
    if len(que) == 0:
        return que
    if isinstance(que[0], int):
        return que
    if isinstance(que[0], list):
        for i in range(len(que)):
            que[i] = _char2num(que[i])
        return que

    for i in range(len(que)):
        que[i] = _RealCard2EnvCard[que[i]]
    return que


def _gen_prompt_bak(info_set):
    dict_data = {
        'landlord_cards': _char2num(info_set.all_handcards["landlord"]),
        'landlord_up_cards': _char2num(info_set.all_handcards["landlord_up"]),
        'landlord_down_cards': _char2num(info_set.all_handcards["landlord_down"]),
        'legal_actions': _char2num(info_set.legal_actions)
    }
    sentence = f"Here is a Peasants vs Landlord card game. Assume you are the landlord. The current cards for the " \
               f"landlord are: {dict_data['landlord_cards']}. The current cards for the peasant that plays before " \
               f"landlord are: {dict_data['landlord_up_cards']}. The current cards for the peasant that plays " \
               f"after landlord are: {dict_data['landlord_down_cards']}. The legal actions for the landlord are: " \
               f"{dict_data['legal_actions']}. Please predict the best action for the landlord."
    return sentence


def _gen_prompt(info_set):
    dict_data = {
        'landlord_cards': _char2num(info_set.all_handcards["landlord"]),
        'landlord_up_cards': _char2num(info_set.all_handcards["landlord_up"]),
        'landlord_down_cards': _char2num(info_set.all_handcards["landlord_down"]),
        'legal_actions': _char2num(info_set.legal_actions),
        'last_two_moves': _char2num(info_set.last_two_moves)
    }
    sentence = f"Here is a Peasants vs Landlord card game. Assume you are the landlord. The current cards for the " \
               f"landlord are: {dict_data['landlord_cards']}. The current cards for the peasant that plays before " \
               f"landlord are: {dict_data['landlord_up_cards']}. The current cards for the peasant that plays " \
               f"after landlord are: {dict_data['landlord_down_cards']}. The last two moves by other players " \
               f"are {dict_data['last_two_moves']}. The legal actions for the landlord are: " \
               f"{dict_data['legal_actions']}. First, you need to select several possible actions from all the " \
               f"legal actions. Then, predict the corresponding responses your opponent might adopt. Finally, " \
               f"based on the prediction results, provide the best action."
    return sentence


def _load_model(position, model_path):
    from douzero.dmc.models import model_dict
    model = model_dict[position]()
    model_state_dict = model.state_dict()
    if torch.cuda.is_available():
        pretrained = torch.load(model_path, map_location='cuda:0')
    else:
        pretrained = torch.load(model_path, map_location='cpu')
    pretrained = {k: v for k, v in pretrained.items() if k in model_state_dict}
    model_state_dict.update(pretrained)
    model.load_state_dict(model_state_dict)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


class LLMAgent:

    def __init__(self, position, model_path, aux):
        self.aux = aux
        # The pretrained model path.
        ckpt = '/mnt/nfs/whl/dozero/train-v3/output_sa/checkpoint-63800/pytorch_model.bin'
        ckpt = '/mnt/nfs/whl/dozero/train-v3/output_sa/checkpoint-5200/pytorch_model.bin'

        ckpt = '/mnt/nfs/whl/dozero/train-v3/output_alltune/checkpoint-32100/pytorch_model.bin'

        # Prepare the lora model
        # base_model = '/mnt/nfs/whl/LLM/llama-2-7b-hf'
        # peft_config = LoraConfig(
        #     task_type=TaskType.CAUSAL_LM, inference_mode=True, r=16, lora_alpha=32, lora_dropout=0
        # )
        # self.tokenizer = LlamaTokenizer.from_pretrained(base_model, trust_remote_code=True)
        # model = LlamaForCausalLM.from_pretrained(base_model, trust_remote_code=True, revision='main', device_map='auto',
        #                                          torch_dtype=torch.float16, load_in_8bit=True)

        # model = prepare_model_for_int8_training(model)
        # model = get_peft_model(model, peft_config)
        # sd = torch.load(ckpt)
        # model.load_state_dict(sd)

        # Prepare the full parameter model
        base_model = "/mnt/nfs/whl/dozero/train-v3/output_alltune/checkpoint-32100"
        self.tokenizer = LlamaTokenizer.from_pretrained("/mnt/nfs/whl/LLM/llama-2-7b-hf", trust_remote_code=True)
        model = LlamaForCausalLM.from_pretrained(base_model, trust_remote_code=True, revision='main', device_map='auto',
                                                 torch_dtype=torch.float16)

        # Resume
        self.model = model
        self.douzero_model = _load_model(position, model_path)
        self.model = self.model.eval()
        self.douzero_model = self.douzero_model.eval()

        global stopping_criteria
        stopping_criteria = StoppingCriteriaList()
        stopping_criteria.append(StopAtSpecificTokenCriteria(self.tokenizer))

    def _get_douzero_action(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        obs = get_obs(infoset)

        z_batch = torch.from_numpy(obs['z_batch']).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()
        if torch.cuda.is_available():
            z_batch, x_batch = z_batch.cuda(), x_batch.cuda()
        y_pred = self.douzero_model.forward(z_batch, x_batch, return_value=True)['values']
        y_pred = y_pred.detach().cpu().numpy()
        print('y_pred', y_pred)

        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]

        return best_action

    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        prompt_raw = _gen_prompt(infoset)
        # The following code is direct inference without tool function.
        prompt = self.tokenizer.encode(
            prompt_raw,
            max_length=1024,
            truncation=True,
            add_special_tokens=True
        )
        prompt = torch.tensor([prompt]).cuda()
        # # Generate the results.
        generate_ids = self.model.generate(inputs=prompt, max_length=4096, do_sample=False)
        generate_ids = generate_ids[:, prompt.shape[1]:]
        answer = \
            self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # The following code is inference using tool function.
        # global INFOSET
        # INFOSET = copy.deepcopy(infoset)
        # answer = tool_generate_txt(self.model, self.tokenizer, prompt_raw)

        try:
            # answer = eval(answer.split(':')[-1].strip())
            pattern = r'(?:Therefore, I will finally play )(.*)'
            answer = eval(re.findall(pattern, answer)[0].strip())
        except:
            answer = []

        if self.aux:
            douzero_answer = self._get_douzero_action(infoset)
            if answer != douzero_answer:
                # warnings.warn(f"The pred answer {answer} does not match douzero action {douzero_answer}."
                #               f" Complete prompt: {prompt_raw}")
                print("Pred False", answer, douzero_answer, prompt_raw)
            else:
                print("Pred True")
            return douzero_answer

        if answer in infoset.legal_actions:
            return answer
        else:
            warnings.warn(f"The pred answer {answer} does not exist in legal actions {infoset.legal_actions}.")
            return infoset.legal_actions[0]

