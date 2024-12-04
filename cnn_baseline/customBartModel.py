from ast import List
from typing import Optional
import torch
from torch import nn
import transformers
from transformers.models.bart.modeling_bart import * # 直接將整個目標全部引入


class CustomBartModel(BartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        # padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # self.encoder = BartEncoder(config, self.shared)
        # self.decoder = BartDecoder(config, self.shared)

        # Initialize weights and apply final processing
        # self.post_init()
        
        # self.decoder = CustomBartDecoder(config, self.shared)
        
        # 创建一个权重分配层
        # self.weighted_sum = nn.Linear(2, 1, bias=False)
        # self.weighted_sum.weight.data = torch.tensor([[0.8, 0.2]])  # new和origin修改权重为0.3和0.7
        
        # 定义权重
        self.weight_more = 0.8#0.7#0.8#0.55#0.65
        self.weight_less = 0.2#0.3#0.2#0.45#0.35

        
        # 创建一个新的解码器层，用于额外的cross-attention
        self.new_decoder_layer = BartDecoderLayer(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_h=None #加入新特征
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None: # 当 encoder_outputs 为None时，会调用 BartModel 中的 self.encoder 进行编码
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput): # 当给定了 encoder_outputs 时，则直接将其封装到 BaseModelOutput 中，并实例化给 encoder_outputs 。猜测这个写法是方便用户使用不同的 encoder？
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        # 分配权重7:3 方法一：在同一层加入sent_h；方法二：可以多加一层在加入sent_h
        #print(encoder_outputs[0].shape, sent_h.shape) # torch.Size([3, 512, 1024]) torch.Size([1, 13, 1024])
        # new_decoder_outputs = self.new_decoder_layer(
        #     encoder_outputs[0],  # 暂时用这个，后期可以换
        #     encoder_hidden_states=sent_h,  # 使用sent_h作为k和v
        #     # encoder_attention_mask=attention_mask,
        #     output_attentions=output_attentions,
        #     # attention_mask=decoder_attention_mask
        # )
        
        # cross_h1 = encoder_outputs[0] # torch.Size([3, 512, 1024])
        # cross_h2 = new_decoder_outputs[0] # torch.Size([3, 512, 1024])
        
        #版本三
        ori = encoder_outputs[0]
        shape = sent_h.shape[1]
        # aa = sent_h.shape   
        sent_h_tmp = sent_h.permute(0,2,1)
        # a = sent_h_tmp.shape
        fc = nn.Linear(shape, 256).to('cuda')
        sent_h_tmp2 = fc(sent_h_tmp)
        # b = sent_h_tmp2.shape
        sent_h = sent_h_tmp2.permute(0,2,1) #问chatgpt是否需要归一化
        
        #对sent_h进行归一化
        # 计算最小值和最大值
        sent_h_min = torch.min(sent_h)
        sent_h_max = torch.max(sent_h)

        # 归一化到(-1, 1)
        sent_h_normalized = 2 * (sent_h - sent_h_min) / (sent_h_max - sent_h_min) - 1
        sent_h = sent_h_normalized
        c = sent_h.shape # (batch_size,256,1024)
        d = encoder_outputs[0].shape
        # cross_h3 = torch.cat((encoder_outputs[0],sent_h),dim=0)
        # a = cross_h3.shape
        cross_h3 = self.weight_more * encoder_outputs[0] + self.weight_less * sent_h
        
        
        #版本二
        # shape = sent_h.shape[1]
        # a = sent_h.shape   
        # sent_h_tmp = sent_h.permute(0,2,1)
        # a = sent_h_tmp.shape
        # fc = nn.Linear(shape, 256).to('cuda')
        # sent_h_tmp2 = fc(sent_h_tmp)
        # b = sent_h_tmp2.shape
        # sent_h = sent_h_tmp2.permute(0,2,1)
        # c = sent_h.shape
        # d = encoder_outputs[0].shape
        # cross_h3 = self.weight_more * encoder_outputs[0] + self.weight_less * sent_h


        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=cross_h3, # 在这个位置加入新特征sent_h
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

# def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         decoder_input_ids=None,
#         decoder_attention_mask=None,
#         head_mask=None,
#         decoder_head_mask=None,
#         cross_attn_head_mask=None,
#         encoder_outputs=None,
#         past_key_values=None,
#         inputs_embeds=None,
#         decoder_inputs_embeds=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):

#         # different to other models, Bart automatically creates decoder_input_ids from
#         # input_ids if no decoder_input_ids are provided
#         if decoder_input_ids is None and decoder_inputs_embeds is None:
#             if input_ids is None:
#                 raise ValueError(
#                     "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
#                     "passed, `input_ids` cannot be `None`. Please pass either "
#                     "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
#                 )

#             decoder_input_ids = shift_tokens_right(
#                 input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
#             )

#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if encoder_outputs is None:
#             encoder_outputs = self.encoder(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 head_mask=head_mask,
#                 inputs_embeds=inputs_embeds,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#             )
#         # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
#         elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
#             encoder_outputs = BaseModelOutput(
#                 last_hidden_state=encoder_outputs[0],
#                 hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
#                 attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
#             )

#         # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
#         decoder_outputs = self.decoder(
#             input_ids=decoder_input_ids,
#             attention_mask=decoder_attention_mask,
#             encoder_hidden_states=encoder_outputs[0],
#             encoder_attention_mask=attention_mask,
#             head_mask=decoder_head_mask,
#             cross_attn_head_mask=cross_attn_head_mask,
#             past_key_values=past_key_values,
#             inputs_embeds=decoder_inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         if not return_dict:
#             return decoder_outputs + encoder_outputs

#         return Seq2SeqModelOutput(
#             last_hidden_state=decoder_outputs.last_hidden_state,
#             past_key_values=decoder_outputs.past_key_values,
#             decoder_hidden_states=decoder_outputs.hidden_states,
#             decoder_attentions=decoder_outputs.attentions,
#             cross_attentions=decoder_outputs.cross_attentions,
#             encoder_last_hidden_state=encoder_outputs.last_hidden_state,
#             encoder_hidden_states=encoder_outputs.hidden_states,
#             encoder_attentions=encoder_outputs.attentions,
#         )
