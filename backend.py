import argparse
import torch

from cumo.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cumo.conversation import conv_templates, SeparatorStyle
from cumo.model.builder import load_pretrained_model
from cumo.utils import disable_torch_init
from cumo.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

from flask import Flask, request, jsonify, redirect, url_for
import os

from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to hold the model and tokenizer
model = None
tokenizer = None
image_processor = None
context_len = None
conv = None
roles = None
image_tensor = None
image_size = None

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

@app.route('/initialize', methods=['POST'])
def initialize():
    global model, tokenizer, image_processor, context_len, conv, roles
    data = request.json
    model_path = data.get('model_path')
    model_base = data.get('model_base')
    conv_mode = data.get('conv_mode', 'mistral_instruct_system')
    device = data.get('device', 'cuda')
    load_8bit = data.get('load_8bit', False)
    load_4bit = data.get('load_4bit', False)

    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, load_8bit, load_4bit, device=device)
    model.config.training = False

    conv = conv_templates[conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
    
    return jsonify({"status": "Model initialized successfully."})

@app.route('/predict', methods=['POST'])
def predict():
    global model, tokenizer, image_processor, context_len, conv, roles, image_tensor, image_size
    if not model:
        return jsonify({"error": "Model not initialized."}), 400
    
    file = request.files.get('file')
    image_url = request.form.get('image_url')
    user_input = request.form.get('input', 'Describe this image.')

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        image = load_image(file_path)
    elif image_url:
        image = load_image(image_url)
    else:
        return jsonify({"error": "No image file or URL provided."}), 400

    image_size = image.size
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    if model.config.mm_use_im_start_end:
        user_input = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + user_input
    else:
        user_input = DEFAULT_IMAGE_TOKEN + '\n' + user_input
    conv.append_message(conv.roles[0], user_input)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True if float(request.form.get('temperature', 0.2)) > 0 else False,
            temperature=float(request.form.get('temperature', 0.2)),
            max_new_tokens=int(request.form.get('max_new_tokens', 512)),
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True)

    outputs = tokenizer.decode(output_ids[0]).strip()
    conv.messages[-1][-1] = outputs

    return jsonify({"response": outputs})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default="mistral_instruct_system")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    
    app.run(host='0.0.0.0', port=args.port)
