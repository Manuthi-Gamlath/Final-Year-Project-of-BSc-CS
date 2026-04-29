import transformers
from copy import deepcopy
from conversation import get_conv_template
from constants import (CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD,
                        IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN,
                        SIGLIP_MEAN, SIGLIP_STD)
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
print("IGNORE_TOKEN_ID",IGNORE_TOKEN_ID)
from transformers import AutoTokenizer
model_path = 'OpenGVLab/InternVideo2_5_Chat_8B'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
data_item ={"id": "video_3827", "video": "video_3827.mp4", "conversations": [{"from": "human", "value": "<image>\nGenerate video descriptions that include details of visual effects, character actions, and movement of people/objects within frames. Describe this video and its style to generate a description. Pay attention to all objects in the video. Do not describe each frame individually. Instead of describing the imaginary content, only describing the content one can determine confidently. Do not describe the contents by itemizing them in list form. "}, {"from": "gpt", "value": "The video captures a sequence of activities centered around a glass-top table in a living room. The scene includes a red and black plaid blanket draped over a piece of furniture, a set of transparent curtains with lace trim hanging in front of a window, and a painting or picture with a red flower on the wall. Various items are spread across the table, including books, a red cloth, a jacket or bag, and a glass tumbler. \n\nA person moves in and out of the frame, interacting with the objects on the table. They rearrange the books, stack them, and adjust other items, like the red cloth, indicating an effort to organize or tidy up. Their movements are gradual and deliberate, categorically focusing on repositioning the objects on the table. \n\nIn the background, idle decorations and framed pictures add to the composition. The video is a straightforward, everyday activity focusing on the reorganization of a living room space, devoid of any dramatic effects or significant changes in the setting."}], "duration": 23.77147423092289}
num_frames=4
num_video_token = 4
num_patches = 4
num_image_tokens = [num_video_token] * num_patches
image_list=["a", "b", "c", "d"]  # Placeholder for image paths
duration = data_item['duration']
msg = f"\nThe video lasts for {duration:.2f} seconds, and {num_frames} frames are uniformly sampled from it. "
# Generate special tokens for each video frame
special_tokens = msg.strip() + '\n' + '\n'.join(['Frame{}: <image>'.format(i + 1) for i in range(len(image_list))])

if '<image>\n' in data_item['conversations'][0]['value']:
    data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
        '<image>\n', special_tokens)
else:
    data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
        '<image>', special_tokens)


def preprocess_internlm(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1,
        model_max_length = None
):
    if model_max_length is None:
        model_max_length = tokenizer.model_max_length

    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    #print("sources",sources)
    for i, source in enumerate(sources):
        #source[0]['from'] is human and source[1]['value'] is the gpt
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        print('1',source)
        print()
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            sentence['value'] = sentence['value'].strip()
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())
        print("conversations",conversations)
        print()

    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
                conversation = conversation.replace('<image>', image_tokens, 1)
            new_conversations.append(conversation)
        conversations = new_conversations

    print("conversations",conversations)

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False if group_by_length or use_packed_ds else 'max_length',
        max_length=model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    print(targets)
    print()
    #print(targets)
    # upto here input is vidon based place holders and user promt only
    # raise ValueError(f'input_ids')

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())  # 浦语里面 pad_token_id = eos_token_id
        #print(total_len) # this give only the input size even the tockernizer is padded 
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID  # <s> # in target start tokern is replaceed with -100
        #print(target)
        parts = conversation.split(conv.roles[1])  # [UNUSED_TOKEN_146]assistant\n
        print(parts[0])
        print()
        print(conv.roles[1])
        info = parts[0] + conv.roles[1]
        
        temp_len = len(tokenizer(info).input_ids) - 1  # 去除tokenizer的<s>
        #print(target.shape,cur_len,cur_len + temp_len)
        target[cur_len: cur_len + temp_len] = IGNORE_TOKEN_ID
        cur_len = cur_len + temp_len
        #print(tokenizer.batch_decode(target, skip_special_tokens=True))
        for index in range(1, len(parts) - 1):
            info = parts[index]
            part1, part2 = info.split(conv.roles[0])
            temp_len = len(tokenizer(part1).input_ids) - 1
            cur_len = cur_len + temp_len
            part = conv.roles[0] + part2 + conv.roles[1]
            temp_len = len(tokenizer(part).input_ids) - 1
            target[cur_len: cur_len + temp_len] = IGNORE_TOKEN_ID
            cur_len = cur_len + temp_len
        last_info = parts[-1]
        temp_len = len(tokenizer(last_info).input_ids) - 1
        cur_len = cur_len + temp_len

        target[cur_len:] = IGNORE_TOKEN_ID
        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            print(repr(tokenizer.decode(z)))

        if cur_len < model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}. This dataset is {ds_name}.')
                sys.stdout.flush()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


ret = preprocess_internlm('internvl2_5', [deepcopy(data_item['conversations'])],
                                  tokenizer, num_image_tokens, group_by_length=True,
                                  ds_name="", num_image=num_patches, model_max_length=5000)

input_ids=ret['input_ids'][0],
labels=ret['labels'][0]
#print(input_ids)
#print(labels)
#print(tokenizer.decode(92542))