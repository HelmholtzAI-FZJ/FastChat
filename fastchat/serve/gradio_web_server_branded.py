"""
The gradio demo server for chatting with a single model.
"""

import argparse
from collections import defaultdict
import datetime
import hashlib
import json
import os
import random
import time
import uuid

import gradio as gr
import requests

from fastchat.constants import (
    LOGDIR,
    WORKER_API_TIMEOUT,
    ErrorCode,
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    RATE_LIMIT_MSG,
    SERVER_ERROR_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
    SESSION_EXPIRATION_TIME,
)
from fastchat.model.model_adapter import (
    get_conversation_template,
)
from fastchat.model.model_registry import get_model_info, model_info
from fastchat.serve.api_provider import get_api_provider_stream_iter
from fastchat.serve.remote_logger import get_remote_logger
from fastchat.utils import (
    build_logger,
    get_window_url_params_js,
    get_window_url_params_with_tos_js,
    moderation_filter,
    parse_gradio_auth_creds,
    load_image,
)

logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "FastChat Client"}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True, visible=True)
disable_btn = gr.Button(interactive=False)
invisible_btn = gr.Button(interactive=False, visible=False)

controller_url = None
enable_moderation = False
use_remote_storage = False

acknowledgment_md = """
<img src="https://helmholtzai-fzj.github.io/FastChat/blablador.png" width="160" alt="Alex Strube\'s dog"><BR>
Made with ❤️ by Helmholtz AI Jülich.<BR>
Get in touch with us at <a href="mailto:blablador@fz-juelich.de">blablador@fz-juelich.de</a>.<BR>
API access (see <a href="https://sdlaml.pages.jsc.fz-juelich.de/ai/guides/blablador_api_access/">documentation</a>) is available too!<BR>
You can also subscribe to our <a href="https://lists.fz-juelich.de/mailman/listinfo/blablador-news">blablador-news</a> mailing list!
"""

# JSON file format of API-based models:
# {
#   "gpt-3.5-turbo": {
#     "model_name": "gpt-3.5-turbo",
#     "api_type": "openai",
#     "api_base": "https://api.openai.com/v1",
#     "api_key": "sk-******",
#     "anony_only": false
#   }
# }
#
#  - "api_type" can be one of the following: openai, anthropic, gemini, or mistral. For custom APIs, add a new type and implement it accordingly.
#  - "anony_only" indicates whether to display this model in anonymous mode only.

api_endpoint_info = {}


class State:
    def __init__(self, model_name, is_vision=False):
        self.conv = get_conversation_template(model_name)
        self.conv_id = uuid.uuid4().hex
        self.skip_next = False
        self.model_name = model_name
        self.oai_thread_id = None
        self.is_vision = is_vision

        # NOTE(chris): This could be sort of a hack since it assumes the user only uploads one image. If they can upload multiple, we should store a list of image hashes.
        self.has_csam_image = False

        self.regen_support = True
        if "browsing" in model_name:
            self.regen_support = False
        self.init_system_prompt(self.conv)

    def init_system_prompt(self, conv):
        system_prompt = conv.get_system_message()
        if len(system_prompt) == 0:
            return
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        system_prompt = system_prompt.replace("{{currentDateTime}}", current_date)
        conv.set_system_message(system_prompt)

    def to_gradio_chatbot(self):
        return self.conv.to_gradio_chatbot()

    def dict(self):
        base = self.conv.dict()
        base.update(
            {
                "conv_id": self.conv_id,
                "model_name": self.model_name,
            }
        )

        if self.is_vision:
            base.update({"has_csam_image": self.has_csam_image})
        return base


def set_global_vars(controller_url_, enable_moderation_, use_remote_storage_):
    global controller_url, enable_moderation, use_remote_storage
    controller_url = controller_url_
    enable_moderation = enable_moderation_
    use_remote_storage = use_remote_storage_


def get_conv_log_filename(is_vision=False, has_csam_image=False):
    t = datetime.datetime.now()
    conv_log_filename = f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json"
    if is_vision and not has_csam_image:
        name = os.path.join(LOGDIR, f"vision-tmp-{conv_log_filename}")
    elif is_vision and has_csam_image:
        name = os.path.join(LOGDIR, f"vision-csam-{conv_log_filename}")
    else:
        name = os.path.join(LOGDIR, conv_log_filename)

    return name


def get_model_list(controller_url, register_api_endpoint_file, vision_arena):
    global api_endpoint_info

    # Add models from the controller
    if controller_url:
        ret = requests.post(controller_url + "/refresh_all_workers")
        assert ret.status_code == 200

        if vision_arena:
            ret = requests.post(controller_url + "/list_multimodal_models")
            models = ret.json()["models"]
        else:
            ret = requests.post(controller_url + "/list_language_models")
            models = ret.json()["models"]
    else:
        models = []

    # Add models from the API providers
    if register_api_endpoint_file:
        api_endpoint_info = json.load(open(register_api_endpoint_file))
        for mdl, mdl_dict in api_endpoint_info.items():
            mdl_vision = mdl_dict.get("vision-arena", False)
            mdl_text = mdl_dict.get("text-arena", True)
            if vision_arena and mdl_vision:
                models.append(mdl)
            if not vision_arena and mdl_text:
                models.append(mdl)

    # Remove anonymous models
    models = list(set(models))
    models_to_remove = ['gpt-3.5-turbo', 'text-davinci-003', 'text-embedding-ada-002']
    visible_models = models.copy()
    for mdl in models:
        if mdl.startswith('alias-') or mdl in models_to_remove:
            visible_models.remove(mdl)
        if mdl not in api_endpoint_info:
            continue
        mdl_dict = api_endpoint_info[mdl]
        if mdl_dict["anony_only"]:
            visible_models.remove(mdl)


    # Sort models and add descriptions
    priority = {k: f"___{i:03d}" for i, k in enumerate(model_info)}
    models.sort(key=lambda x: priority.get(x, x))
    visible_models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"All models: {models}")
    logger.info(f"Visible models: {visible_models}")
    return visible_models, models


def load_demo_single(models, url_params):
    selected_model = models[0] if len(models) > 0 else ""
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            selected_model = model

    dropdown_update = gr.Dropdown(choices=models, value=selected_model, visible=True)
    state = None
    return state, dropdown_update


def load_demo(url_params, request: gr.Request):
    global models

    ip = get_ip(request)
    logger.info(f"load_demo. ip: {ip}. params: {url_params}")

    if args.model_list_mode == "reload":
        models, all_models = get_model_list(
            controller_url, args.register_api_endpoint_file, vision_arena=False
        )

    return load_demo_single(models, url_params)


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    filename = get_conv_log_filename()
    if "llava" in model_selector:
        filename = filename.replace("2024", "vision-tmp-2024")

    with open(filename, "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
    get_remote_logger().log(data)


def upvote_last_response(state, model_selector, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"upvote. ip: {ip}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"downvote. ip: {ip}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"flag. ip: {ip}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"regenerate. ip: {ip}")
    if not state.regen_support:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5
    state.conv.update_last_message(None)
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def clear_history(request: gr.Request):
    ip = get_ip(request)
    logger.info(f"clear_history. ip: {ip}")
    state = None
    return (state, [], "", None) + (disable_btn,) * 5


def get_ip(request: gr.Request):
    if "cf-connecting-ip" in request.headers:
        ip = request.headers["cf-connecting-ip"]
    elif "x-forwarded-for" in request.headers:
        ip = request.headers["x-forwarded-for"]
    else:
        ip = request.client.host
    return ip


# TODO(Chris): At some point, we would like this to be a live-reporting feature.
def report_csam_image(state, image):
    pass


def _prepare_text_with_image(state, text, images, csam_flag):
    if images is not None and len(images) > 0:
        image = images[0]

        if len(state.conv.get_images()) > 0:
            # reset convo with new image
            state.conv = get_conversation_template(state.model_name)

        image = state.conv.convert_image_to_base64(
            image
        )  # PIL type is not JSON serializable

        if csam_flag:
            state.has_csam_image = True
            report_csam_image(state, image)

        text = text, [image]

    return text


def add_text(state, model_selector, text, image, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"add_text. ip: {ip}. len: {len(text)}")

    if state is None:
        state = State(model_selector)

    if len(text) <= 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5

    all_conv_text = state.conv.get_prompt()
    all_conv_text = all_conv_text[-2000:] + "\nuser: " + text
    flagged = moderation_filter(all_conv_text, [state.model_name])
    # flagged = moderation_filter(text, [state.model_name])
    if flagged:
        logger.info(f"violate moderation. ip: {ip}. text: {text}")
        # overwrite the original text
        text = MODERATION_MSG

    if (len(state.conv.messages) - state.conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {ip}. text: {text}")
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), CONVERSATION_LIMIT_MSG, None) + (
            no_change_btn,
        ) * 5

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    text = _prepare_text_with_image(state, text, image, csam_flag=False)
    state.conv.append_message(state.conv.roles[0], text)
    state.conv.append_message(state.conv.roles[1], None)
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def model_worker_stream_iter(
    conv,
    model_name,
    worker_addr,
    prompt,
    temperature,
    repetition_penalty,
    top_p,
    max_new_tokens,
    images,
):
    # Make requests
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "stop": conv.stop_str,
        "stop_token_ids": conv.stop_token_ids,
        "echo": False,
    }

    logger.info(f"==== request ====\n{gen_params}")

    if len(images) > 0:
        gen_params["images"] = images

    # Stream output
    response = requests.post(
        worker_addr + "/worker_generate_stream",
        headers=headers,
        json=gen_params,
        stream=True,
        timeout=WORKER_API_TIMEOUT,
    )
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode())
            yield data


def is_limit_reached(model_name, ip):
    monitor_url = "http://localhost:9090"
    try:
        ret = requests.get(
            f"{monitor_url}/is_limit_reached?model={model_name}&user_id={ip}", timeout=1
        )
        obj = ret.json()
        return obj
    except Exception as e:
        logger.info(f"monitor error: {e}")
        return None


def bot_response(
    state,
    temperature,
    top_p,
    max_new_tokens,
    request: gr.Request,
    apply_rate_limit=True,
    use_recommended_config=False,
):
    ip = get_ip(request)
    logger.info(f"bot_response. ip: {ip}")
    start_tstamp = time.time()
    temperature = float(temperature)
    top_p = float(top_p)
    max_new_tokens = int(max_new_tokens)

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        state.skip_next = False
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if apply_rate_limit:
        ret = is_limit_reached(state.model_name, ip)
        if ret is not None and ret["is_limit_reached"]:
            error_msg = RATE_LIMIT_MSG + "\n\n" + ret["reason"]
            logger.info(f"rate limit reached. ip: {ip}. error_msg: {ret['reason']}")
            state.conv.update_last_message(error_msg)
            yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
            return

    conv, model_name = state.conv, state.model_name
    model_api_dict = (
        api_endpoint_info[model_name] if model_name in api_endpoint_info else None
    )
    images = conv.get_images()

    if model_api_dict is None:
        # Query worker address
        ret = requests.post(
            controller_url + "/get_worker_address", json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

        # No available worker
        if worker_addr == "":
            conv.update_last_message(SERVER_ERROR_MSG)
            yield (
                state,
                state.to_gradio_chatbot(),
                disable_btn,
                disable_btn,
                disable_btn,
                enable_btn,
                enable_btn,
            )
            return

        # Construct prompt.
        # We need to call it here, so it will not be affected by "▌".
        prompt = conv.get_prompt()
        # Set repetition_penalty
        if "t5" in model_name:
            repetition_penalty = 1.2
        else:
            repetition_penalty = 1.0

        stream_iter = model_worker_stream_iter(
            conv,
            model_name,
            worker_addr,
            prompt,
            temperature,
            repetition_penalty,
            top_p,
            max_new_tokens,
            images,
        )
    else:
        if use_recommended_config:
            recommended_config = model_api_dict.get("recommended_config", None)
            if recommended_config is not None:
                temperature = recommended_config.get("temperature", temperature)
                top_p = recommended_config.get("top_p", top_p)
                max_new_tokens = recommended_config.get(
                    "max_new_tokens", max_new_tokens
                )

        stream_iter = get_api_provider_stream_iter(
            conv,
            model_name,
            model_api_dict,
            temperature,
            top_p,
            max_new_tokens,
            state,
        )

    html_code = ' <span class="cursor"></span> '

    # conv.update_last_message("▌")
    conv.update_last_message(html_code)
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        data = {"text": ""}
        for i, data in enumerate(stream_iter):
            if data["error_code"] == 0:
                output = data["text"].strip()
                # conv.update_last_message(output + "▌")
                conv.update_last_message(output + html_code)
                yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
            else:
                output = data["text"] + f"\n\n(error_code: {data['error_code']})"
                conv.update_last_message(output)
                yield (state, state.to_gradio_chatbot()) + (
                    disable_btn,
                    disable_btn,
                    disable_btn,
                    enable_btn,
                    enable_btn,
                )
                return
        output = data["text"].strip()
        conv.update_last_message(output)
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5
    except requests.exceptions.RequestException as e:
        conv.update_last_message(
            f"{SERVER_ERROR_MSG}\n\n"
            f"(error_code: {ErrorCode.GRADIO_REQUEST_ERROR}, {e})"
        )
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return
    except Exception as e:
        conv.update_last_message(
            f"{SERVER_ERROR_MSG}\n\n"
            f"(error_code: {ErrorCode.GRADIO_STREAM_UNKNOWN_ERROR}, {e})"
        )
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    finish_tstamp = time.time()
    logger.info(f"{output}")

    conv.save_new_images(
        has_csam_images=state.has_csam_image, use_remote_storage=use_remote_storage
    )

    filename = get_conv_log_filename(
        is_vision=state.is_vision, has_csam_image=state.has_csam_image
    )

    with open(filename, "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "gen_params": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
            },
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
    get_remote_logger().log(data)


block_css = """
#notice_markdown .prose {
    font-size: 110% !important;
}
#notice_markdown th {
    display: none;
}
#notice_markdown td {
    padding-top: 6px;
    padding-bottom: 6px;
}
#arena_leaderboard_dataframe table {
    font-size: 110%;
}
#full_leaderboard_dataframe table {
    font-size: 110%;
}
#model_description_markdown {
    font-size: 110% !important;
}
#leaderboard_markdown .prose {
    font-size: 110% !important;
}
#leaderboard_markdown td {
    padding-top: 6px;
    padding-bottom: 6px;
}
#leaderboard_dataframe td {
    line-height: 0.1em;
}
#about_markdown .prose {
    font-size: 110% !important;
}
#ack_markdown .prose {
    font-size: 110% !important;
}
#chatbot .prose {
    font-size: 105% !important;
}
.sponsor-image-about img {
    margin: 0 20px;
    margin-top: 20px;
    height: 40px;
    max-height: 100%;
    width: auto;
    float: left;
}

.chatbot h1, h2, h3 {
    margin-top: 8px; /* Adjust the value as needed */
    margin-bottom: 0px; /* Adjust the value as needed */
    padding-bottom: 0px;
}

.chatbot h1 {
    font-size: 130%;
}
.chatbot h2 {
    font-size: 120%;
}
.chatbot h3 {
    font-size: 110%;
}
.chatbot p:not(:first-child) {
    margin-top: 8px;
}

.typing {
    display: inline-block;
}

.cursor {
    display: inline-block;
    width: 7px;
    height: 1em;
    background-color: black;
    vertical-align: middle;
    animation: blink 1s infinite;
}

.dark .cursor {
    display: inline-block;
    width: 7px;
    height: 1em;
    background-color: white;
    vertical-align: middle;
    animation: blink 1s infinite;
}

@keyframes blink {
    0%, 50% { opacity: 1; }
    50.1%, 100% { opacity: 0; }
}

.app {
  max-width: 100% !important;
  padding: 20px !important;               
}

a {
    color: #1976D2; /* Your current link color, a shade of blue */
    text-decoration: none; /* Removes underline from links */
}
a:hover {
    color: #63A4FF; /* This can be any color you choose for hover */
    text-decoration: underline; /* Adds underline on hover */
}

footer {visibility: hidden}
"""


def get_model_description_md(models):
    model_description_md = """
| | | |
| ---- | ---- | ---- |
"""
    ct = 0
    visited = set()
    for i, name in enumerate(models):
        minfo = get_model_info(name)
        if minfo.simple_name in visited:
            continue
        visited.add(minfo.simple_name)
        one_model_md = f"[{minfo.simple_name}]({minfo.link}): {minfo.description}"

        if ct % 3 == 0:
            model_description_md += "|"
        model_description_md += f" {one_model_md} |"
        if ct % 3 == 2:
            model_description_md += "\n"
        ct += 1
    return model_description_md


def build_about():
    about_markdown = """
# About Us
Chatbot Arena is an open-source research project developed by members from [LMSYS](https://lmsys.org) and UC Berkeley [SkyLab](https://sky.cs.berkeley.edu/). Our mission is to build an open platform to evaluate LLMs by human preference in the real-world.
We open-source our [FastChat](https://github.com/lm-sys/FastChat) project at GitHub and release chat and human feedback dataset. We invite everyone to join us!

## Arena Core Team
- [Lianmin Zheng](https://lmzheng.net/) (co-lead), [Wei-Lin Chiang](https://infwinston.github.io/) (co-lead), [Ying Sheng](https://sites.google.com/view/yingsheng/home), [Joseph E. Gonzalez](https://people.eecs.berkeley.edu/~jegonzal/), [Ion Stoica](http://people.eecs.berkeley.edu/~istoica/)

## Past Members
- [Siyuan Zhuang](https://scholar.google.com/citations?user=KSZmI5EAAAAJ), [Hao Zhang](https://cseweb.ucsd.edu/~haozhang/)

## Learn more
- Chatbot Arena [paper](https://arxiv.org/abs/2403.04132), [launch blog](https://lmsys.org/blog/2023-05-03-arena/), [dataset](https://github.com/lm-sys/FastChat/blob/main/docs/dataset_release.md), [policy](https://lmsys.org/blog/2024-03-01-policy/)
- LMSYS-Chat-1M dataset [paper](https://arxiv.org/abs/2309.11998), LLM Judge [paper](https://arxiv.org/abs/2306.05685)

## Contact Us
- Follow our [X](https://x.com/lmsysorg), [Discord](https://discord.gg/HSWAKCrnFx) or email us at lmsys.org@gmail.com
- File issues on [GitHub](https://github.com/lm-sys/FastChat)
- Download our datasets and models on [HuggingFace](https://huggingface.co/lmsys)

## Acknowledgment
We thank [SkyPilot](https://github.com/skypilot-org/skypilot) and [Gradio](https://github.com/gradio-app/gradio) team for their system support.
We also thank [UC Berkeley SkyLab](https://sky.cs.berkeley.edu/), [Kaggle](https://www.kaggle.com/), [MBZUAI](https://mbzuai.ac.ae/), [a16z](https://www.a16z.com/), [Together AI](https://www.together.ai/), [Hyperbolic](https://hyperbolic.xyz/), [Anyscale](https://www.anyscale.com/), [HuggingFace](https://huggingface.co/) for their generous sponsorship. Learn more about partnership [here](https://lmsys.org/donations/).

<div class="sponsor-image-about">
    <img src="https://storage.googleapis.com/public-arena-asset/skylab.png" alt="SkyLab">
    <img src="https://storage.googleapis.com/public-arena-asset/kaggle.png" alt="Kaggle">
    <img src="https://storage.googleapis.com/public-arena-asset/mbzuai.jpeg" alt="MBZUAI">
    <img src="https://storage.googleapis.com/public-arena-asset/a16z.jpeg" alt="a16z">
    <img src="https://storage.googleapis.com/public-arena-asset/together.png" alt="Together AI">
    <img src="https://storage.googleapis.com/public-arena-asset/hyperbolic_logo.png" alt="Hyperbolic">
    <img src="https://storage.googleapis.com/public-arena-asset/anyscale.png" alt="AnyScale">
    <img src="https://storage.googleapis.com/public-arena-asset/huggingface.png" alt="HuggingFace">
</div>
"""
    gr.Markdown(about_markdown, elem_id="about_markdown")


def build_single_model_ui(models, add_promotion_links=True):
    blablador_logo = (
    '''
    <svg class="svg-hh-ai-logo" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 458 36" fill="none" aria-labelledby="helmholtz-ai-logo-header" role="img">
									<title id="helmholtz-logo-ai-header">Helmholtz AI Logo</title>
									<g clip-path="url(#a)">
										<path class="svg-hh-ai-logo__letters" fill="#005aa0" d="M16.304 1.73h5.833v30.588h-5.833V19.984H5.881v12.334H.048V1.729H5.88v12.374h10.423V1.729Zm15.238 0c-.265 0-.516.132-.688.305l-1.892 1.903c-.211.173-.304.426-.304.692v24.814c0 .266.093.519.304.692l1.892 1.902c.172.213.423.306.688.306h16.296v-5.867H34.492V20.01h11.587v-5.867H34.492V7.637h13.346V1.729H31.542ZM56.62 32.303h13.557v-5.867H59.57V1.729h-5.833V29.43c0 .267.08.52.304.692l1.892 1.903c.172.213.423.306.688.306v-.027Zm57.154-18.201V1.729h-5.833v30.589h5.833V19.984h10.423v12.334h5.834V1.729h-5.834v12.374h-10.423Zm38.796 9.487c0 .266-.133.479-.304.691l-1.839 1.85a1.02 1.02 0 0 1-.728.306h-5.238c-.264 0-.476-.093-.687-.306l-1.839-1.85c-.212-.212-.304-.425-.304-.691V10.484c0-.266.092-.519.304-.692l1.839-1.903c.211-.173.423-.306.687-.306h5.238c.265 0 .516.133.728.306l1.839 1.903c.171.173.304.426.304.692v13.119-.014Zm1.375-20.397c-.939-.905-2.142-1.464-3.439-1.464h-6.812c-1.283 0-2.526.519-3.478 1.464l-3.003 3.02a4.908 4.908 0 0 0-1.415 3.446v14.716c0 1.29.515 2.54 1.415 3.459l3.003 3.06c.899.945 2.182 1.424 3.478 1.424h6.812a4.906 4.906 0 0 0 3.439-1.424l3.043-3.06a4.92 4.92 0 0 0 1.415-3.46V9.66c0-1.29-.516-2.541-1.415-3.446l-3.043-3.02Zm12.976 29.111h13.809v-5.867h-10.846V1.729h-5.833V29.43c0 .267.079.52.304.692l1.892 1.903a.857.857 0 0 0 .687.306l-.013-.027Zm20.383-24.707v24.72h5.834V7.598h8.068V1.729H179.25v5.868h8.069-.014Zm34.775-5.575a1.004 1.004 0 0 0-.688-.306h-17.024v5.867h12.05l-12.605 20.065v1.77c0 .265.132.518.304.691l1.892 1.903a.856.856 0 0 0 .687.306h17.407V26.45h-12.605l12.777-20.317V4.616a1.04 1.04 0 0 0-.304-.692l-1.891-1.902ZM95.614 1.729l-8.108 14.13-8.108-14.13h-5.582v30.589h5.833v-18.92l5.238 9.114a.96.96 0 0 0 .847.492h3.571a.977.977 0 0 0 .847-.492l5.185-9.048v18.84h5.833V1.73h-5.556ZM293.956 1.09l4.048 12.36v.294h-1.786l-.926-2.954h-4.748l-.926 2.954h-1.786v-.293l4.048-12.36h2.076Zm-1.045 2.063-1.865 6h3.744l-1.879-6ZM305.107 4.177v1.584h-1.376c-.952 0-1.706.439-2.143 1.21v6.773h-1.719V4.177h1.547l.08.785c.608-.519 1.415-.785 2.314-.785h1.31-.013ZM309.631 1.61v2.567h3.121v1.584h-3.121v5.189c0 .785.383 1.21 1.137 1.21h1.535v1.584h-1.588c-1.825 0-2.804-.985-2.804-2.741V5.76h-2.513V4.177h2.037c.397 0 .516-.173.516-.519V1.61h1.693-.013ZM314.022 1.53c0-.719.516-1.198 1.19-1.198.675 0 1.191.48 1.191 1.198 0 .678-.503 1.17-1.191 1.17-.687 0-1.19-.505-1.19-1.17Zm2.05 12.214h-1.719V4.177h1.719v9.567ZM321.681 4.177h3.121v1.584h-3.121v7.983h-1.72V5.76h-2.434V4.177h2.434V3.02c0-1.757.979-2.701 2.738-2.701h1.442v1.583h-1.323c-.727 0-1.124.4-1.124 1.21v1.065h-.013ZM326.032 1.53c0-.719.516-1.198 1.191-1.198.674 0 1.19.48 1.19 1.198 0 .678-.502 1.17-1.19 1.17s-1.191-.505-1.191-1.17Zm2.05 12.214h-1.719V4.177h1.719v9.567ZM330.411 10.098V7.823c0-2.329 1.613-3.872 4.047-3.872 2.302 0 3.915 1.517 3.915 3.659v.213h-1.719V7.61c0-1.198-.9-2.022-2.183-2.022-1.441 0-2.301.864-2.301 2.275v2.195c0 1.41.886 2.275 2.301 2.275 1.283 0 2.183-.811 2.183-1.969v-.213h1.719v.213c0 2.156-1.547 3.606-3.915 3.606-2.367 0-4.047-1.544-4.047-3.872ZM340.278 1.53c0-.719.516-1.198 1.19-1.198.675 0 1.191.48 1.191 1.198 0 .678-.503 1.17-1.191 1.17-.688 0-1.19-.505-1.19-1.17Zm2.05 12.214h-1.719V4.177h1.719v9.567ZM352.156 7.517v6.227h-1.548l-.053-.759c-.674.639-1.626.985-2.804.985-2.09 0-3.359-1.038-3.359-2.754 0-1.557.992-2.448 3.121-2.741l2.897-.386V7.49c0-1.237-.714-1.902-2.077-1.902-1.362 0-2.129.678-2.129 1.796v.133h-1.746v-.133c0-2.076 1.508-3.433 3.915-3.433s3.77 1.29 3.77 3.566h.013Zm-1.733 3.579V9.459l-2.606.386c-1.164.16-1.693.639-1.693 1.344 0 .785.675 1.277 1.746 1.277 1.138 0 2.077-.505 2.553-1.37ZM354.709 13.744V.319h1.72v13.425h-1.72ZM366.402 1.09v12.654h-1.786V1.09h1.786ZM376.93 7.663v6.067h-1.746V7.796c0-1.344-.806-2.222-2.063-2.222-.992 0-1.825.48-2.262 1.238v6.918h-1.719V4.177h1.547l.08.865c.648-.692 1.574-1.104 2.685-1.104 2.142 0 3.492 1.423 3.492 3.725h-.014ZM382.116 1.61v2.567h3.121v1.584h-3.121v5.189c0 .785.383 1.21 1.137 1.21h1.534v1.584H383.2c-1.825 0-2.804-.985-2.804-2.741V5.76h-2.513V4.177h2.037c.397 0 .516-.173.516-.519V1.61h1.693-.013ZM394.073 9.592h-6.243v.466c0 1.424.886 2.289 2.341 2.289 1.244 0 2.169-.692 2.183-1.597h1.719c-.092 1.903-1.666 3.22-3.915 3.22-2.447 0-4.061-1.517-4.061-3.859V7.823c0-2.329 1.614-3.872 4.021-3.872 2.408 0 3.955 1.543 3.955 3.859v1.796-.014Zm-6.243-1.716v.266h4.524v-.266c0-1.424-.86-2.315-2.223-2.315-1.362 0-2.301.905-2.301 2.315ZM396.375 13.744V.319h1.719v13.425h-1.719ZM400.845 13.744V.319h1.72v13.425h-1.72ZM404.985 1.53c0-.719.516-1.198 1.191-1.198.674 0 1.19.48 1.19 1.198 0 .678-.502 1.17-1.19 1.17s-1.191-.505-1.191-1.17Zm2.051 12.214h-1.72V4.177h1.72v9.567ZM415.779 4.177h1.534v9.487c0 2.235-1.534 3.592-4.021 3.592s-3.968-1.29-3.968-3.34v-.12h1.759v.12c0 1.065.86 1.757 2.262 1.757 1.402 0 2.248-.732 2.248-1.943v-1.423c-.648.598-1.547.944-2.605.944-2.223 0-3.638-1.45-3.638-3.738v-1.77c0-2.315 1.415-3.779 3.664-3.779 1.111 0 2.05.386 2.698 1.038l.08-.811-.013-.014Zm-.199 6.24V6.812c-.436-.772-1.283-1.238-2.262-1.238-1.402 0-2.262.852-2.262 2.222V9.42c0 1.37.86 2.222 2.262 2.222.992 0 1.826-.465 2.262-1.21v-.014ZM427.485 9.592h-6.243v.466c0 1.424.886 2.289 2.341 2.289 1.243 0 2.169-.692 2.182-1.597h1.72c-.093 1.903-1.667 3.22-3.915 3.22-2.447 0-4.061-1.517-4.061-3.859V7.823c0-2.329 1.614-3.872 4.021-3.872s3.942 1.543 3.942 3.859v1.796l.013-.014Zm-6.243-1.716v.266h4.523v-.266c0-1.424-.859-2.315-2.222-2.315-1.362 0-2.301.905-2.301 2.315ZM437.498 7.663v6.067h-1.746V7.796c0-1.344-.807-2.222-2.077-2.222-.992 0-1.825.48-2.262 1.238v6.918h-1.719V4.177h1.547l.08.865c.648-.692 1.574-1.104 2.685-1.104 2.143 0 3.492 1.423 3.492 3.725ZM439.614 10.098V7.823c0-2.329 1.614-3.872 4.048-3.872 2.301 0 3.915 1.517 3.915 3.659v.213h-1.72V7.61c0-1.198-.899-2.022-2.182-2.022-1.442 0-2.302.864-2.302 2.275v2.195c0 1.41.887 2.275 2.302 2.275 1.283 0 2.182-.811 2.182-1.969v-.213h1.72v.213c0 2.156-1.548 3.606-3.915 3.606-2.368 0-4.048-1.544-4.048-3.872ZM457.365 9.592h-6.243v.466c0 1.424.886 2.289 2.341 2.289 1.244 0 2.169-.692 2.183-1.597h1.719c-.092 1.903-1.666 3.22-3.915 3.22-2.447 0-4.061-1.517-4.061-3.859V7.823c0-2.329 1.614-3.872 4.021-3.872 2.408 0 3.942 1.543 3.942 3.859v1.796l.013-.014Zm-6.243-1.716v.266h4.524v-.266c0-1.424-.86-2.315-2.223-2.315-1.362 0-2.301.905-2.301 2.315ZM288.507 27.834v-3.779c0-2.794 1.931-4.683 4.828-4.683 2.897 0 4.748 1.836 4.748 4.47v.466h-1.785v-.466c0-1.756-1.111-2.794-2.99-2.794-1.878 0-3.002 1.118-3.002 3.06v3.646c0 1.942 1.098 3.047 3.002 3.047 1.905 0 2.99-1.038 2.99-2.794v-.48h1.785v.48c0 2.647-1.957 4.47-4.748 4.47-2.791 0-4.828-1.876-4.828-4.643ZM300.067 28.605v-2.261c0-2.355 1.574-3.899 4.008-3.899s4.048 1.544 4.048 3.899v2.261c0 2.329-1.588 3.872-4.048 3.872s-4.008-1.516-4.008-3.871Zm6.323-.04V26.41c0-1.424-.86-2.315-2.302-2.315-1.441 0-2.301.892-2.301 2.315v2.156c0 1.41.899 2.275 2.301 2.275 1.403 0 2.302-.892 2.302-2.275ZM310.001 28.605v-2.261c0-2.355 1.574-3.899 4.008-3.899s4.047 1.544 4.047 3.899v2.261c0 2.329-1.587 3.872-4.047 3.872s-4.008-1.516-4.008-3.871Zm6.323-.04V26.41c0-1.424-.86-2.315-2.302-2.315-1.442 0-2.301.892-2.301 2.315v2.156c0 1.41.899 2.275 2.301 2.275 1.402 0 2.302-.892 2.302-2.275ZM328.241 26.237V28.7c0 2.275-1.442 3.778-3.664 3.778-1.058 0-1.957-.346-2.606-.958v4.005h-1.719v-12.84h1.548l.079.812c.674-.678 1.614-1.037 2.698-1.037 2.222 0 3.664 1.49 3.664 3.778Zm-1.746.08c0-1.344-.86-2.222-2.248-2.222-.992 0-1.839.479-2.276 1.277v4.218c.424.772 1.244 1.25 2.276 1.25 1.375 0 2.248-.85 2.248-2.221v-2.315.013ZM338.069 28.113h-6.243v.466c0 1.424.886 2.288 2.341 2.288 1.243 0 2.169-.692 2.182-1.596h1.72c-.093 1.902-1.667 3.22-3.915 3.22-2.448 0-4.061-1.517-4.061-3.859v-2.288c0-2.329 1.613-3.872 4.021-3.872 2.407 0 3.942 1.543 3.942 3.858v1.797l.013-.014Zm-6.243-1.716v.266h4.523v-.266c0-1.424-.86-2.315-2.222-2.315s-2.301.904-2.301 2.315ZM345.622 22.698v1.583h-1.376c-.952 0-1.706.44-2.143 1.211v6.772h-1.719v-9.566h1.547l.08.785c.608-.519 1.415-.785 2.315-.785h1.309-.013ZM354.193 26.024v6.227h-1.547l-.053-.758c-.675.638-1.627.984-2.805.984-2.089 0-3.359-1.037-3.359-2.754 0-1.557.992-2.448 3.121-2.74l2.897-.387v-.598c0-1.238-.714-1.903-2.077-1.903-1.362 0-2.129.679-2.129 1.796v.133h-1.746v-.133c0-2.075 1.508-3.432 3.915-3.432s3.77 1.29 3.77 3.565h.013Zm-1.746 3.593V27.98l-2.606.386c-1.164.16-1.693.639-1.693 1.344 0 .785.675 1.277 1.746 1.277 1.138 0 2.077-.505 2.553-1.37ZM359.391 20.13v2.568h3.122v1.583h-3.122v5.19c0 .784.384 1.21 1.138 1.21h1.534v1.583h-1.587c-1.825 0-2.804-.984-2.804-2.74V24.28h-2.513v-1.583h2.037c.397 0 .516-.173.516-.519V20.13h1.693-.014ZM363.783 20.05c0-.718.516-1.197 1.19-1.197.675 0 1.191.479 1.191 1.197 0 .679-.503 1.171-1.191 1.171-.688 0-1.19-.505-1.19-1.17Zm2.05 12.201h-1.72v-9.566h1.72v9.566ZM368.147 28.605v-2.261c0-2.355 1.574-3.899 4.008-3.899s4.048 1.544 4.048 3.899v2.261c0 2.329-1.587 3.872-4.048 3.872-2.46 0-4.008-1.516-4.008-3.871Zm6.323-.04V26.41c0-1.424-.86-2.315-2.301-2.315-1.442 0-2.302.892-2.302 2.315v2.156c0 1.41.899 2.275 2.302 2.275 1.402 0 2.301-.892 2.301-2.275ZM386.203 26.184v6.067h-1.746v-5.934c0-1.344-.807-2.222-2.063-2.222-.993 0-1.826.479-2.262 1.237v6.92h-1.72v-9.567h1.548l.079.864c.648-.691 1.574-1.104 2.685-1.104 2.143 0 3.492 1.424 3.492 3.726l-.013.013ZM394.046 28.166v-8.555h1.799v8.555c0 1.677.887 2.662 2.487 2.662 1.601 0 2.526-.959 2.526-2.662v-8.555h1.799v8.555c0 2.648-1.653 4.325-4.325 4.325s-4.299-1.677-4.299-4.325h.013ZM413.014 26.184v6.067h-1.745v-5.934c0-1.344-.807-2.222-2.077-2.222-.992 0-1.826.479-2.262 1.237v6.92h-1.72v-9.567h1.548l.079.864c.649-.691 1.574-1.104 2.686-1.104 2.142 0 3.491 1.424 3.491 3.726v.013ZM415.236 20.05c0-.718.516-1.197 1.191-1.197.674 0 1.19.479 1.19 1.197 0 .679-.502 1.171-1.19 1.171s-1.191-.505-1.191-1.17Zm2.051 12.201h-1.72v-9.566h1.72v9.566ZM422.988 20.13v2.568h3.121v1.583h-3.121v5.19c0 .784.383 1.21 1.124 1.21h1.534v1.583h-1.587c-1.825 0-2.804-.984-2.804-2.74V24.28h-2.513v-1.583h2.037c.396 0 .516-.173.516-.519V20.13h1.693Z"></path> \
										<path classe="svg-hh-ai-logo__signet" fill="#00B1EB" d="m238.495 26.357-2.104 5.947h-6.177l10.463-29.896c.132-.426.476-.692.9-.692h5.105c.384 0 .767.253.9.652l10.462 29.936h-6.177l-2.103-5.947h-11.283.014Zm2.063-5.868h7.156l-3.598-10.311-3.558 10.311ZM266.854 32.304h-5.833V1.73h5.833v30.589-.014Z"></path> \
									</g></svg>
    '''
)
    blablador_logo = gr.HTML(blablador_logo)

    notice_markdown = f"""
# This is _*BLABLADOR*_, our experimental large language model server! 🐕‍🦺
### Different models might be available at Alex Strube's whim. These are the models currently running:
"""

    state = gr.State()
    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Group(elem_id="share-region-named"):
        with gr.Row(elem_id="model_selector_row"):
            model_selector = gr.Dropdown(
                choices=models,
                value=models[0] if len(models) > 0 else "",
                interactive=True,
                show_label=False,
                container=False,
            )
        # with gr.Row():
        #     with gr.Accordion(
        #         f"🔍 Expand to see the descriptions of {len(models)} models",
        #         open=False,
        #     ):
        #         model_description_md = get_model_description_md(models)
        #         gr.Markdown(model_description_md, elem_id="model_description_markdown")

        chatbot = gr.Chatbot(
            elem_id="chatbot",
            label="This is what I have to say.... Remember: I am a BLABLADOR! Not all I say is true or even real",
            height=300,
            scale=2,
            show_copy_button=True,
        )
    with gr.Row():
        textbox = gr.Textbox(
            show_label=False,
            placeholder="Enter your question and press ENTER",
            elem_id="input_box",
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=0)

    with gr.Row() as button_row:
        # upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
        # downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
        # flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    with gr.Accordion("Parameters", open=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=32768,
            value=1024,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    if add_promotion_links:
        gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    # Register listeners
    imagebox = gr.State(None)
    btn_list = [regenerate_btn, clear_btn]
    # upvote_btn.click(
    #     upvote_last_response,
    #     [state, model_selector],
    #     [textbox, upvote_btn, downvote_btn, flag_btn],
    # )
    # downvote_btn.click(
    #     downvote_last_response,
    #     [state, model_selector],
    #     [textbox, upvote_btn, downvote_btn, flag_btn],
    # )
    # flag_btn.click(
    #     flag_last_response,
    #     [state, model_selector],
    #     [textbox, upvote_btn, downvote_btn, flag_btn],
    # )
    regenerate_btn.click(
        regenerate, state, [state, chatbot, textbox, imagebox] + btn_list
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    clear_btn.click(clear_history, None, [state, chatbot, textbox, imagebox] + btn_list)

    model_selector.change(
        clear_history, None, [state, chatbot, textbox, imagebox] + btn_list
    )

    textbox.submit(
        add_text,
        [state, model_selector, textbox, imagebox],
        [state, chatbot, textbox, imagebox] + btn_list,
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    send_btn.click(
        add_text,
        [state, model_selector, textbox, imagebox],
        [state, chatbot, textbox, imagebox] + btn_list,
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )

    return [state, model_selector]


def build_demo(models):
    with gr.Blocks(
        title="BLABLADOR - The experimental Helmholtz AI LLM server",
        theme=gr.themes.Base(),
        css=block_css,
    ) as demo:
        url_params = gr.JSON(visible=False)

        state, model_selector = build_single_model_ui(models)

        if args.model_list_mode not in ["once", "reload"]:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

        if args.show_terms_of_use:
            load_js = get_window_url_params_with_tos_js
        else:
            load_js = get_window_url_params_js

        demo.load(
            load_demo,
            [url_params],
            [
                state,
                model_selector,
            ],
            js=load_js,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to generate a public, shareable link",
    )
    parser.add_argument(
        "--controller-url",
        type=str,
        default="http://localhost:21001",
        help="The address of the controller",
    )
    parser.add_argument(
        "--concurrency-count",
        type=int,
        default=10,
        help="The concurrency count of the gradio queue",
    )
    parser.add_argument(
        "--model-list-mode",
        type=str,
        default="once",
        choices=["once", "reload"],
        help="Whether to load the model list once or reload the model list every time",
    )
    parser.add_argument(
        "--moderate",
        action="store_true",
        help="Enable content moderation to block unsafe inputs",
    )
    parser.add_argument(
        "--show-terms-of-use",
        action="store_true",
        help="Shows term of use before loading the demo",
    )
    parser.add_argument(
        "--register-api-endpoint-file",
        type=str,
        help="Register API-based model endpoints from a JSON file",
    )
    parser.add_argument(
        "--gradio-auth-path",
        type=str,
        help='Set the gradio authentication file path. The file should contain one or more user:password pairs in this format: "u1:p1,u2:p2,u3:p3"',
    )
    parser.add_argument(
        "--gradio-root-path",
        type=str,
        help="Sets the gradio root path, eg /abc/def. Useful when running behind a reverse-proxy or at a custom URL path prefix",
    )
    parser.add_argument(
        "--use-remote-storage",
        action="store_true",
        default=False,
        help="Uploads image files to google cloud storage if set to true",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # Set global variables
    set_global_vars(args.controller_url, args.moderate, args.use_remote_storage)
    models, all_models = get_model_list(
        args.controller_url, args.register_api_endpoint_file, vision_arena=False
    )

    # Set authorization credentials
    auth = None
    if args.gradio_auth_path is not None:
        auth = parse_gradio_auth_creds(args.gradio_auth_path)

    # Launch the demo
    demo = build_demo(models)
    demo.queue(
        default_concurrency_limit=args.concurrency_count,
        status_update_rate=10,
        api_open=False,
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=200,
        auth=auth,
        root_path=args.gradio_root_path,
    )
