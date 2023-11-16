from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, CLIPImageProcessor

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..valley_arch import ValleyVideoMetaModel, ValleyVideoMetaForCausalLM, ValleyProductMetaModel, ValleyProductMetaForCausalLM

from valley.util.data_util import load_video, preprocess_multimodal, KeywordsStoppingCriteria, tokenizer_image_token

from valley import conversation as conversation_lib

from PIL import Image

class ValleyConfig(LlamaConfig):
    model_type = "valley"


class ValleyVideoLlamaModel(ValleyVideoMetaModel, LlamaModel):
    config_class = ValleyConfig

    def __init__(self, config: LlamaConfig):
        super(ValleyVideoLlamaModel, self).__init__(config)


class ValleyVideoLlamaForCausalLM(LlamaForCausalLM, ValleyVideoMetaForCausalLM):
    config_class = ValleyConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = ValleyVideoLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def build_inputs(self, tokenizer, messages, num_image=1, image_token_len = 224,if_context=False, conv = None):
        tokenizer.padding_side = 'left'
        prompt = ''
        sources = messages.copy()
        for sentence in sources:
            sentence['value'] = sentence['content']
            sentence.pop('content')
        
        messages = preprocess_multimodal([sources],{'is_multimodal': True})[0]
        
        roles = {"user": conv.roles[0], "assistant": conv.roles[1]}

        for i, message in enumerate(messages):
            if message["role"] == 'system':
                conv.system = message["value"]
                messages = messages[1:]
                break

        conv.messages = []
        if conv.sep_style == conversation_lib.SeparatorStyle.PLAIN:
            for j, sentence in enumerate(messages):
                role = roles[sentence["role"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
        else:
            for j, sentence in enumerate(messages):
                role = roles[sentence["role"]]
                conv.append_message(role, sentence["value"])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print(prompt)
        input_id = tokenizer_image_token(prompt, tokenizer, return_tensors='pt', image_token_len = image_token_len, num_image = num_image)
        return input_id
    
    def process_response(self,outputs):
        output = []
        for i, out in enumerate(outputs):
            while True:
                cur_len = len(out)
                out = out.strip()
                for pattern in ['###', 'Assistant:', 'Response:', 'Valley:']:
                    if out.startswith(pattern):
                        out = out[len(pattern):].strip()
                if len(out) == cur_len:
                    break
            try:
                index = out.index('###')
            except ValueError:
                out += '###'
                index = out.index("###")
            out = out[:index].strip()
            output.append(out)
        return output

    @torch.no_grad()
    def completion(self, tokenizer, video: str, image: str ,message: list, gen_kwargs:dict, device, frame_mode='fixed',fps=0.5,fixed_frame_number=8, conv_mode = 'v1'):
        if video:
            images = load_video(video, frame_mode=frame_mode, fps_number= fps, fixed_frame_number= fixed_frame_number)
            images = images.permute(1, 0, 2, 3)
            images = images.unsqueeze(0).half().to(device)
            print(images.shape)
        elif image:
            if isinstance(image, list) and isinstance(image[0], str):
                image = [Image.open(img) for img in image]
            elif isinstance(image, list) and not isinstance(image[0], str):
                image = [img for img in image]
            elif isinstance(image, str):
                image = [Image.open(image)]
            else:
                image = [image]
            processor_config = {"crop_size": 224,
                                "do_center_crop": True,
                                "do_normalize": True,
                                "do_resize": True,
                                "feature_extractor_type": "CLIPFeatureExtractor",
                                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                                "image_std":  [0.26862954, 0.26130258,0.27577711],
                                "resample": 3,
                                "size": 224
                                }
            image_processer = CLIPImageProcessor(**processor_config)
            images = image_processer.preprocess(
                image, return_tensors='pt')['pixel_values'].unsqueeze(0).half().to(device)
            # images = images.permute(1, 0, 2, 3)
            # print(images.shape)
        else:
            images = None

        conv = conversation_lib.conv_templates[conv_mode].copy()
        
        inputs = self.build_inputs(tokenizer, message, images.shape[1] if images is not None else 1, image_token_len = (images.shape[-1]//14)**2, if_context= isinstance(image, list), conv = conv)
        input_ids = inputs.unsqueeze(0).to(device)
        
        stop_str = conv.sep if conv.sep_style != conversation_lib.SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
        output_ids = self.generate(input_ids = input_ids, images = images, stopping_criteria=[stopping_criteria],**gen_kwargs)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        # print(outputs)
        response = self.process_response(outputs)
        return response
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs


class ValleyProductLlamaModel(ValleyProductMetaModel, LlamaModel):
    config_class = ValleyConfig

    def __init__(self, config: LlamaConfig):
        super(ValleyProductLlamaModel, self).__init__(config)


class ValleyProductLlamaForCausalLM(LlamaForCausalLM, ValleyProductMetaForCausalLM):
    config_class = ValleyConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = ValleyProductLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def build_inputs(self, tokenizer, messages, num_image=1, image_token_len = 224,if_context=False, conv = None):
        tokenizer.padding_side = 'left'
        prompt = ''
        sources = messages.copy()
        for sentence in sources:
            sentence['value'] = sentence['content']
            sentence.pop('content')
        
        messages = preprocess_multimodal([sources],{'is_multimodal': True})[0]
        
        roles = {"user": conv.roles[0], "assistant": conv.roles[1]}

        for i, message in enumerate(messages):
            if message["role"] == 'system':
                conv.system = message["value"]
                messages = messages[1:]
                break

        conv.messages = []
        if conv.sep_style == conversation_lib.SeparatorStyle.PLAIN:
            for j, sentence in enumerate(messages):
                role = roles[sentence["role"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
        else:
            for j, sentence in enumerate(messages):
                role = roles[sentence["role"]]
                conv.append_message(role, sentence["value"])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print(prompt)
        input_id = tokenizer_image_token(prompt, tokenizer, return_tensors='pt', image_token_len = image_token_len, num_image = num_image)
        return input_id
    
    def process_response(self,outputs):
        output = []
        for i, out in enumerate(outputs):
            while True:
                cur_len = len(out)
                out = out.strip()
                for pattern in ['###', 'Assistant:', 'Response:', 'Valley:']:
                    if out.startswith(pattern):
                        out = out[len(pattern):].strip()
                if len(out) == cur_len:
                    break
            try:
                index = out.index('###')
            except ValueError:
                out += '###'
                index = out.index("###")
            out = out[:index].strip()
            output.append(out)
        return output

    @torch.no_grad()
    def completion(self, tokenizer, video: str, image: str ,message: list, gen_kwargs:dict, device, frame_mode='fixed',fps=0.5,fixed_frame_number=8, conv_mode = 'v1'):
        if video:
            images = load_video(video, frame_mode=frame_mode, fps_number= fps, fixed_frame_number= fixed_frame_number)
            images = images.permute(1, 0, 2, 3)
            images = images.unsqueeze(0).half().to(device)
            print(images.shape)
        elif image:
            if isinstance(image, list) and isinstance(image[0], str):
                image = [Image.open(img) for img in image]
            elif isinstance(image, list) and not isinstance(image[0], str):
                image = [img for img in image]
            elif isinstance(image, str):
                image = [Image.open(image)]
            else:
                image = [image]
            processor_config = {"crop_size": 224,
                                "do_center_crop": True,
                                "do_normalize": True,
                                "do_resize": True,
                                "feature_extractor_type": "CLIPFeatureExtractor",
                                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                                "image_std":  [0.26862954, 0.26130258,0.27577711],
                                "resample": 3,
                                "size": 224
                                }
            image_processer = CLIPImageProcessor(**processor_config)
            images = image_processer.preprocess(
                image, return_tensors='pt')['pixel_values'].unsqueeze(0).half().to(device)
            # images = images.permute(1, 0, 2, 3)
            # print(images.shape)
        else:
            images = None

        conv = conversation_lib.conv_templates[conv_mode].copy()
        
        inputs = self.build_inputs(tokenizer, message, images.shape[1] if images is not None else 1, image_token_len = (images.shape[-1]//14)**2, if_context= isinstance(image, list), conv = conv)
        input_ids = inputs.unsqueeze(0).to(device)
        
        stop_str = conv.sep if conv.sep_style != conversation_lib.SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
        output_ids = self.generate(input_ids = input_ids, images = images, stopping_criteria=[stopping_criteria],**gen_kwargs)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        # print(outputs)
        response = self.process_response(outputs)
        return response
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs




AutoConfig.register("valley", ValleyConfig)
AutoModelForCausalLM.register(ValleyConfig, ValleyVideoLlamaForCausalLM)
AutoModelForCausalLM.register(ValleyConfig, ValleyProductLlamaForCausalLM)
