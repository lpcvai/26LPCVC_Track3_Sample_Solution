# This script runs on Linux with native adb commands
import argparse
import glob
import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
import sys
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor

from qwen_vl_utils import process_vision_info
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

script_start_time = time.time()
print("Script execution started...")

# Take the following command line arguments:
# host_output_dir: directory on host to save final outputs defaults to "Host_Outputs/"
# uploads_dir: directory containing contestant's uploaded files (defaults to "contestant_uploads/")
parser = argparse.ArgumentParser(description="Inference Script for Qwen2-VL 2B IoT Challenge")
parser.add_argument("--host_output_dir", type=str, default="Host_Outputs/", help="Directory on host to save final outputs")
parser.add_argument("--uploads_dir", type=str, default="contestant_uploads/", help="Directory containing contestant's uploaded files")
args = parser.parse_args()

# Check and initialize paths to files in contestant_uploads/
# contestant_uploads/
# ├── ar*-ar*-cl*
# │   └── weight_sharing_model_*_of_*.serialized.bin
# ├── embedding_weights*.raw
# ├── inputs.json
# ├── mask.raw
# ├── position_ids_cos.raw
# ├── position_ids_sin.raw
# ├── serialized_binaries
# │   └── veg.serialized.bin
# └── tokenizer.json

execution_ws = os.getcwd()
context_path = execution_ws + "/" + args.uploads_dir
missing = []

# ---- Find ar*-ar*-cl* folders ----
ar_folders = glob.glob(os.path.join(context_path, "ar*-ar*-cl*"))
ar_folders = [f for f in ar_folders if os.path.isdir(f)]

if len(ar_folders) == 0:
    missing.append("ar*-ar*-cl* folder")
elif len(ar_folders) > 1:
    raise RuntimeError(
        f"Multiple ar*-ar*-cl* folders found: "
        f"{[os.path.basename(f) for f in ar_folders]}. "
        "Exactly one is required."
    )
else:
    ar_foldername = os.path.basename(ar_folders[0])

# ---- Check .serialized.bin(s) inside ar* folder ----
ar_bins = glob.glob(os.path.join(context_path, ar_foldername, "*.serialized.bin"))
if not ar_bins:
    missing.append("*.serialized.bin inside ar*-ar*-cl* folder")

# ---- Check embedding_weights*.raw ----
embed_files = glob.glob(os.path.join(context_path, "embedding_weights*.raw"))
embed_weights_filename = embed_files[0] if embed_files else None

if embed_weights_filename is None:
    missing.append("embedding_weights*.raw")
else:
    embed_weights_filename = os.path.basename(embed_weights_filename)

# ---- Required raw files ----
required_raws = [
    "mask.raw",
    "position_ids_cos.raw",
    "position_ids_sin.raw"
]

for raw in required_raws:
    if not os.path.isfile(os.path.join(context_path, raw)):
        missing.append(raw)

# ---- Check serialized_binaries folder ----
serialized_dir = os.path.join(context_path, "serialized_binaries")
if not os.path.isdir(serialized_dir):
    missing.append("serialized_binaries/")
else:
    serialized_bins = glob.glob(os.path.join(serialized_dir, "*.serialized.bin"))
    if not serialized_bins:
        missing.append("*.serialized.bin inside serialized_binaries/")

# ---- Check JSON files ----
required_jsons = ["inputs.json", "tokenizer.json"]
for jf in required_jsons:
    if not os.path.isfile(os.path.join(context_path, jf)):
        missing.append(jf)

# ---- Final validation ----
if missing:
    print("Missing required files/folders:")
    for m in missing:
        print(f"  - {m}")
    raise RuntimeError("Validation failed: required contestant_uploads content missing")

print("Contestant's upload files passed validation")

inputs_json_path = os.path.join(context_path, "inputs.json")

with open(inputs_json_path, "r") as f:
    inputs = json.load(f)

# ---- Top-level inputs ----
qwen2_vl_processor_input = inputs["qwen_vl_processor"]
llm_config_input = inputs["llm_config"]
inp_h_input = inputs["data_preprocess_inp_h"]
inp_w_input = inputs["data_preprocess_inp_w"]
run_veg_n_tokens_input  = inputs["run_veg_n_tokens"]
run_veg_embedding_dim_input  = inputs["run_veg_embedding_dim"]
# ---- Genie config ----
genie_config = inputs["genie_config"]

ADB = "adb"
cmd = f'{ADB} devices -l'  # command as a single string
result = subprocess.run(cmd, capture_output=True, text=True, shell=True)

lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
device_lines = lines[1:]  # Everything after the header

if not device_lines:
    raise RuntimeError("Error: No devices found.")

device_id = device_lines[0].split()[0]
print(f"Successfully connected to: {device_id}")

sys.path.append('../../')
# Set up NSP Target for GEN5
class AndroidTarget:
    def __init__(self, soc_id, dsp_arch, qnn_htp_lib):
        self.soc_id = soc_id
        self.dsp_arch = dsp_arch
        self.qnn_htp_lib_name = qnn_htp_lib

ANDROID_GEN5 = AndroidTarget(
    soc_id=None,
    dsp_arch="v81",
    qnn_htp_lib="QnnHtpV81"
)
nsp_target = ANDROID_GEN5

# Configure QNN SDK and Genie paths
if os.path.isdir("qnn_assets"):
    os.unlink("qnn_assets")
os.symlink("/qnn", "qnn_assets")

execution_ws = os.getcwd()
QNN_SDK_dir = os.path.join(execution_ws, "qnn_assets")
QNN_lib_dir = os.path.join(QNN_SDK_dir, "lib/aarch64-android")
QNN_binary = os.path.join(QNN_SDK_dir, "bin/aarch64-android/qnn-net-run")
GENIE_lib_dir = os.path.join(QNN_SDK_dir, "lib/aarch64-android")
GENIE_binary = os.path.join(QNN_SDK_dir, "bin/aarch64-android/genie-t2t-run")
QNN_skel = os.path.join(QNN_SDK_dir, "lib/hexagon-" + nsp_target.dsp_arch,  "unsigned", "lib" + nsp_target.qnn_htp_lib_name + "Skel.so")

des_dir = os.path.join(execution_ws, "to_device")
des_dir_models = os.path.join(des_dir, "models")
des_dir_qwen2_models = os.path.join(des_dir_models, "qwen2-vl")
des_dir_qwen2_models_2B = os.path.join(des_dir_qwen2_models, "2B-FT")
des_dir_qwen2_model_2B_data = os.path.join(des_dir_qwen2_models_2B, "data")

if os.path.exists(des_dir):
    shutil.rmtree(des_dir) # clear destination dir 
os.makedirs(des_dir_qwen2_model_2B_data)

target_device_dir = "/data/local/tmp/qwen2_vl_assets"

qwen2_models_context_path = os.path.join(context_path, ar_foldername)

llm_model_names = os.listdir(qwen2_models_context_path)
llm_model_names.sort()
llm_model_names = [f for f in llm_model_names if os.path.isfile(os.path.join(qwen2_models_context_path, f))]

veg_models_context_path = os.path.join(context_path + "/serialized_binaries")

for model_bin in os.listdir(veg_models_context_path):
    src_file = os.path.join(veg_models_context_path, model_bin)
    if os.path.isfile(src_file):
        shutil.copy(src_file, des_dir)

for model_bin in os.listdir(qwen2_models_context_path):
    src_file = os.path.join(qwen2_models_context_path, model_bin)
    if os.path.isfile(src_file):
        shutil.copy(src_file, des_dir_qwen2_models_2B)

qwen2_vl_embedding_buffer_file = os.path.join(context_path, embed_weights_filename)
shutil.copy(qwen2_vl_embedding_buffer_file, des_dir_qwen2_models_2B)

QNN_libs = ["libQnnHtp.so", "libQnnHtpNetRunExtensions.so", "libQnnHtpPrepare.so", "lib" + nsp_target.qnn_htp_lib_name + "Stub.so", "libQnnSystem.so"]
for lib in QNN_libs:
    shutil.copy(os.path.join(QNN_lib_dir, lib), des_dir)

GENIE_libs = ["libGenie.so"]
for lib in GENIE_libs:
    shutil.copy(os.path.join(GENIE_lib_dir, lib), des_dir)

shutil.copy(QNN_binary, des_dir)
shutil.copy(GENIE_binary, des_dir)
shutil.copy(QNN_skel, des_dir)

qwen2_vl_tokenizer_path = context_path + "/tokenizer.json"
shutil.copy(qwen2_vl_tokenizer_path, des_dir_qwen2_models)

qwen2_data_folder_rel_path = os.path.relpath(des_dir_qwen2_model_2B_data, des_dir)
target_device_data_dir = os.path.join(target_device_dir, qwen2_data_folder_rel_path)

htp_backend_extensions_data = {
    "backend_extensions": {
        "shared_library_path": "libQnnHtpNetRunExtensions.so",
        "config_file_path": os.path.join(target_device_data_dir, "htp_backend_ext_config.json")
    }
}

htp_backend_ext_config_data = {
    "devices": [
        {
            "cores":[{
                "perf_profile": "burst",
                "rpc_control_latency": 100
            }]
        }
    ]
}

with open(os.path.join(des_dir, 'htp_backend_extensions.json'),'w') as f:
    f.write(json.dumps(htp_backend_extensions_data, indent=4))
with open(os.path.join(des_dir_qwen2_model_2B_data,  'htp_backend_ext_config.json'),'w') as f:
    f.write(json.dumps(htp_backend_ext_config_data, indent=4))

dialog = genie_config["dialog"]
dialog["tokenizer"]["path"] = str("models/qwen2-vl/tokenizer.json")
dialog["engine"]["backend"]["extensions"] = str("models/qwen2-vl/2B-FT/data/htp_backend_ext_config.json")

BASE_MODEL_DIR = Path("models/qwen2-vl/2B-FT")
dialog["engine"]["model"]["binary"]["ctx-bins"] = [
    str(BASE_MODEL_DIR / name) for name in llm_model_names
]

with open(os.path.join(des_dir, 'qwen2-vl-e2t-htp.json'), 'w') as f:
    f.write(json.dumps(genie_config, indent=4))

cmd_rm = [ADB, "-s", device_id, "shell", "rm", "-rf", target_device_dir]
result_rm = subprocess.run(cmd_rm, capture_output=True, text=True)
print(result_rm.stdout, result_rm.stderr)

# Batch files
zip_path = os.path.join(os.path.dirname(des_dir), "package.zip")
device_zip_path = f"{target_device_dir}/package.zip"

if os.path.exists(zip_path):
    os.remove(zip_path)

print("Zipping directory:", des_dir)
t0 = time.time()
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED) as zipf:
    for root, dirs, files in os.walk(des_dir):
        for file in files:
            full_path = os.path.join(root, file)
            arcname = os.path.relpath(full_path, des_dir)
            zipf.write(full_path, arcname)
t1 = time.time()
print(f"\tZipping completed in {t1 - t0:.2f}s")

print("Pushing ZIP to device...")
cmd_push = [ADB, "-s", device_id, "push", zip_path, device_zip_path]
t2 = time.time()
subprocess.run(cmd_push, capture_output=True, text=True)
t3 = time.time()
print(f"adb push time: {t3 - t2:.2f}s\n")

print("Unzipping on device...")
cmd_unzip = f'{ADB} -s {device_id} shell "cd {target_device_dir} && unzip -o package.zip"'
t4 = time.time()
subprocess.run(cmd_unzip, shell=True, capture_output=True, text=True)
t5 = time.time()
print(f"\tadb unzip time: {t5 - t4:.2f}s")

print("Removing ZIPs from device and host...")
subprocess.run([ADB, "-s", device_id, "shell", "rm", "-f", device_zip_path], capture_output=True, text=True)
os.remove(zip_path)

lookup_table_np = np.fromfile(os.path.join(des_dir_qwen2_models_2B, embed_weights_filename), dtype=np.float32)
lookup_table_np = lookup_table_np.reshape(genie_config["dialog"]["context"]["n-vocab"], genie_config["dialog"]["embedding"]["size"])

def get_embeddings(token_ids):
    token_embeddings =  []
    for token_id in token_ids:
        token_embeddings.append(lookup_table_np[token_id, :])
    token_embeddings_np = np.stack(token_embeddings, axis=0)
    return token_embeddings_np

qwen2_vl_processor = AutoProcessor.from_pretrained(qwen2_vl_processor_input)

def data_preprocess(processor, img_path, inp_h=342, inp_w=512, prompt=''):
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": img_path,
                "resized_height": inp_h,
                "resized_width": inp_w,
            },
            {
                "type": "text",
                "text": prompt
            },
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        return_tensors="pt",
    )
    return inputs

def text_preprocess(processor, prompt=''):
    """Preprocess text for Stage 2 without image inputs"""
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt
            },
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=text,
        return_tensors="pt",
    )
    return inputs

def run_qnn_net_run(model_context, input_data_list):
    tmp_dirpath = os.path.abspath('tmp/inputs')
    os.makedirs(tmp_dirpath, exist_ok=True)
    
    input_list_text = ''
    for index, input_data in enumerate(input_data_list):
        raw_file_path = f'{tmp_dirpath}/input_{index}.raw'
        input_data.tofile(raw_file_path)
        input_list_text += target_device_dir + '/inputs/' + os.path.basename(raw_file_path) + ' '

    cos_data  = os.path.join(context_path, "position_ids_cos.raw")
    sin_data  = os.path.join(context_path, "position_ids_sin.raw")
    mask_data = os.path.join(context_path, "mask.raw")
    shutil.copy(cos_data, tmp_dirpath)
    shutil.copy(sin_data, tmp_dirpath)
    shutil.copy(mask_data, tmp_dirpath)
    input_list_text += target_device_dir + '/inputs/position_ids_cos.raw' + ' '
    input_list_text += target_device_dir + '/inputs/position_ids_sin.raw' + ' '
    input_list_text += target_device_dir + '/inputs/mask.raw' + ' '

    input_list_filepath = f'{tmp_dirpath}/../input_list.txt'
    with open(input_list_filepath, 'w') as f:
        f.write(input_list_text)

    subprocess.run([ADB, "-s", device_id, "push", input_list_filepath, target_device_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    subprocess.run([ADB, "-s", device_id, "push", tmp_dirpath, target_device_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    subprocess.run(
        [
            ADB, "-s", device_id, "shell",
            f"LD_LIBRARY_PATH={target_device_dir} ADSP_LIBRARY_PATH={target_device_dir} "
            f"{target_device_dir}/qnn-net-run --retrieve_context {model_context} "
            f"--backend {target_device_dir}/libQnnHtp.so --input_list {target_device_dir}/input_list.txt "
            f"--output_dir {target_device_dir} --config_file {target_device_dir}/htp_backend_extensions.json"
        ],
        stdout=open(f"{tmp_dirpath}/log.txt", "w"),
        stderr=subprocess.STDOUT, check=True
    )

    subprocess.run([ADB, "-s", device_id, "pull", f"{target_device_dir}/Result_0/vision_embedding.raw", tmp_dirpath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    output_data = np.fromfile(f'{tmp_dirpath}/vision_embedding.raw', dtype=np.float32)
    return output_data

def run_veg(pixel_values, n_tokens=216, embedding_dim=1536):
    input_data_list = [pixel_values]
    output_data = run_qnn_net_run(f'{target_device_dir}/veg.serialized.bin', input_data_list)
    output_data = output_data.reshape((1, n_tokens, embedding_dim))
    return output_data


# --- Configuration for 2-Stage Process ---
host_image_folder = "dataset/images"
host_prompts_folder = "dataset/prompts"
host_output_dir = Path(args.host_output_dir)
if host_output_dir.exists():
    shutil.rmtree(host_output_dir)
host_output_dir.mkdir(parents=True)

llm_config = AutoConfig.from_pretrained(llm_config_input, trust_remote_code=True) 

# Directories for Stage 1 and Stage 2 on Device
device_embeds_dir_s1 = f"{target_device_dir}/ImageEmbeds_S1"
device_output_dir_s1 = "Outputs_S1" 
device_embeds_dir_s2 = f"{target_device_dir}/ImageEmbeds_S2"
device_output_dir_s2 = "Outputs_S2" 

# Read the Stage 1 and Stage 2 fixed prompts
with open(os.path.join(host_prompts_folder, "stage1.txt"), "r", encoding="utf-8") as f:
    stage1_prompt_text = f.read().strip()
with open(os.path.join(host_prompts_folder, "stage2.txt"), "r", encoding="utf-8") as f:
    stage2_prompt_text = f.read().strip()

# Create Local Temp Structure
TMP_ROOT = Path("tmp")
if TMP_ROOT.exists():
    shutil.rmtree(TMP_ROOT)

TMP_S1 = TMP_ROOT / "stage1"
TMP_S2 = TMP_ROOT / "stage2"
HOST_OUT_S1 = host_output_dir / "stage1"
TMP_S1.mkdir(parents=True, exist_ok=True)
TMP_S2.mkdir(parents=True, exist_ok=True)
HOST_OUT_S1.mkdir(parents=True, exist_ok=True)
# Prepare device directories
subprocess.run(f'{ADB} -s {device_id} shell "mkdir -p {device_embeds_dir_s1} {device_embeds_dir_s2}"', shell=True, check=True)

# Map all images dynamically tracking their subfolder relative paths
image_files = glob.glob(os.path.join(host_image_folder, "**", "*.png"), recursive=True)
print(f"Found {len(image_files)} images for the 2-stage inference.")

image_tasks = []
for img_path in image_files:
    rel_path = os.path.relpath(img_path, host_image_folder)
    safe_name = rel_path.replace('/', '_').replace('\\', '_').replace('.png', '')
    image_tasks.append({
        'img_path': img_path,
        'rel_path': rel_path,
        'safe_name': safe_name
    })

# ==========================================
# STAGE 1: Process Image + Stage1 Text
# ==========================================
print("\n--- Starting Stage 1 Prep ---")
for task in image_tasks:
    try:
        # Preprocess Image + Stage1 prompt
        inputs = data_preprocess(qwen2_vl_processor, task['img_path'], inp_h_input, inp_w_input, stage1_prompt_text)
        pixel_values = inputs['pixel_values'].detach().numpy().astype(np.float32)
        
        image_embeddings_raw = run_veg(pixel_values)
        image_embeddings_torch = torch.from_numpy(image_embeddings_raw)

        token_ids = inputs['input_ids']
        inputs_embeds = torch.from_numpy(get_embeddings(token_ids))

        image_mask = ((inputs['input_ids'] == llm_config.image_token_id)
                      .unsqueeze(-1)
                      .expand_as(inputs_embeds))

        final_embeds = inputs_embeds.masked_scatter(image_mask, image_embeddings_torch).detach().numpy()
        
        raw_path = TMP_S1 / f"{task['safe_name']}.raw"
        final_embeds.tofile(raw_path)
    except Exception as e:
        print(f"Failed Stage 1 prep for {task['safe_name']}: {e}")

# Zip and push S1 embeds
zip_path_s1 = TMP_ROOT / "stage1_embeds.zip"
with zipfile.ZipFile(zip_path_s1, "w", compression=zipfile.ZIP_STORED) as zf:
    for raw_file in TMP_S1.glob("*.raw"):
        zf.write(raw_file, arcname=raw_file.name)

subprocess.run(f'{ADB} -s {device_id} push {zip_path_s1} {device_embeds_dir_s1}/', shell=True, check=True)
subprocess.run(f'{ADB} -s {device_id} shell "cd {device_embeds_dir_s1} && unzip -o {zip_path_s1.name}"', shell=True, stdout=subprocess.DEVNULL)

# Stage 1 Execution Script
script_s1_name = "batch_run_s1.sh"
batch_script_s1 = f"""#!/bin/sh
cd {target_device_dir}
export LD_LIBRARY_PATH={target_device_dir}
export ADSP_LIBRARY_PATH={target_device_dir}

rm -rf {device_output_dir_s1}
mkdir -p {device_output_dir_s1}

# Initialize the shared timing_summary.txt
echo "Timing Summary" > timing_summary.txt

count=0
for f in {device_embeds_dir_s1}/*.raw; do
    [ -e "$f" ] || continue
    filename=$(basename "$f")
    output_name="${{filename%.*}}.txt"
    
    echo "------------------------------------------------" >> timing_summary.txt
    echo "Stage 1 - $filename" >> timing_summary.txt
    
    {{ time ./genie-t2t-run \\
        -c qwen2-vl-e2t-htp.json \\
        -e "$f" \\
        -t models/qwen2-vl/2B-FT/{embed_weights_filename} \\
        | tail -n +5 > "{device_output_dir_s1}/$output_name"; }} 2>> timing_summary.txt
    
    count=$((count + 1))
done
echo "\\nStage 1 processing complete. Processed $count files."
"""
with open(script_s1_name, "w", newline='\n') as f:
    f.write(batch_script_s1)

subprocess.run(f'{ADB} -s {device_id} push {script_s1_name} {target_device_dir}/', shell=True, check=True)
subprocess.run(f'{ADB} -s {device_id} shell "chmod +x {target_device_dir}/{script_s1_name}"', shell=True, check=True)

print("Starting Stage 1 continuous batch inference on device...")
host_start_time = time.time()
subprocess.run(f'{ADB} -s {device_id} shell "{target_device_dir}/{script_s1_name}"', shell=True)
print("Pulling Stage 1 inference outputs...")
subprocess.run(
    f'{ADB} -s {device_id} pull {target_device_dir}/{device_output_dir_s1}/. {HOST_OUT_S1}/', 
    shell=True, 
    check=True
)

# ==========================================
# STAGE 2: Process Stage1 Text + Stage2 Text
# ==========================================
print("\n--- Starting Stage 2 Prep ---")
for task in image_tasks:
    s1_out_file = HOST_OUT_S1 / f"{task['safe_name']}.txt"
    
    if not s1_out_file.exists():
        print(f"Skipping {task['safe_name']} - Stage 1 output not found.")
        continue
        
    with open(s1_out_file, "r", encoding="utf-8") as f:
        s1_raw_text = f.read().strip()

    # Remove [BEGIN]: and [END] tags if they exist to prevent confusion
    s1_clean_text = s1_raw_text.replace("[BEGIN]:", "").replace("[END]", "").strip()
    
    combined_prompt = (
        f"*** INSTRUCTIONS ***\n{stage2_prompt_text}\n\n"
        f"*** ANALYSIS DATA TO PROCESS ***\n{s1_clean_text}\n\n"
    )
    try:
        # Use text_preprocess since Stage 2 has no images
        inputs = text_preprocess(qwen2_vl_processor, combined_prompt)
        token_ids = inputs['input_ids']
        
        # Verify tokens were generated
        if token_ids.shape[1] == 0:
            print(f"Error: No tokens generated for {task['safe_name']}")
            continue

        final_embeds = torch.from_numpy(get_embeddings(token_ids)).detach().numpy()
        
        raw_path = TMP_S2 / f"{task['safe_name']}.raw"
        final_embeds.tofile(raw_path)
    except Exception as e:
        print(f"Failed Stage 2 prep for {task['safe_name']}: {e}")

# Zip and push S2 embeds
zip_path_s2 = TMP_ROOT / "stage2_embeds.zip"
with zipfile.ZipFile(zip_path_s2, "w", compression=zipfile.ZIP_STORED) as zf:
    for raw_file in TMP_S2.glob("*.raw"):
        zf.write(raw_file, arcname=raw_file.name)

subprocess.run(f'{ADB} -s {device_id} push {zip_path_s2} {device_embeds_dir_s2}/', shell=True, check=True)
subprocess.run(f'{ADB} -s {device_id} shell "cd {device_embeds_dir_s2} && unzip -o {zip_path_s2.name}"', shell=True, stdout=subprocess.DEVNULL)

# Stage 2 Execution Script
script_s2_name = "batch_run_s2.sh"
batch_script_s2 = f"""#!/bin/sh
cd {target_device_dir}
export LD_LIBRARY_PATH={target_device_dir}
export ADSP_LIBRARY_PATH={target_device_dir}

rm -rf {device_output_dir_s2}
mkdir -p {device_output_dir_s2}

count=0
for f in {device_embeds_dir_s2}/*.raw; do
    [ -e "$f" ] || continue
    filename=$(basename "$f")
    output_name="${{filename%.*}}.txt"
    
    # Append to the timing summary created in Stage 1
    echo "------------------------------------------------" >> timing_summary.txt
    echo "Stage 2 - $filename" >> timing_summary.txt
    
    {{ time ./genie-t2t-run \\
        -c qwen2-vl-e2t-htp.json \\
        -e "$f" \\
        -t models/qwen2-vl/2B-FT/{embed_weights_filename} \\
        | tail -n +5 > "{device_output_dir_s2}/$output_name"; }} 2>> timing_summary.txt
    
    count=$((count + 1))
done
echo "\\nStage 2 processing complete. Processed $count files."
"""
with open(script_s2_name, "w", newline='\n') as f:
    f.write(batch_script_s2)

subprocess.run(f'{ADB} -s {device_id} push {script_s2_name} {target_device_dir}/', shell=True, check=True)
subprocess.run(f'{ADB} -s {device_id} shell "chmod +x {target_device_dir}/{script_s2_name}"', shell=True, check=True)

print("Starting Stage 2 continuous batch inference on device...")
subprocess.run(f'{ADB} -s {device_id} shell "{target_device_dir}/{script_s2_name}"', shell=True)

# --- Pull Final Results and Format Host_Outputs ---
print("\nPulling Stage 2 Final inference outputs...")
HOST_OUT_S2 = TMP_ROOT / "Outputs_S2"
HOST_OUT_S2.mkdir(parents=True, exist_ok=True)
subprocess.run(f'{ADB} -s {device_id} pull {target_device_dir}/{device_output_dir_s2}/. {HOST_OUT_S2}/', shell=True, check=True)

for task in image_tasks:
    s2_out_file = HOST_OUT_S2 / f"{task['safe_name']}.txt"
    if s2_out_file.exists():
        final_rel_path = task['rel_path'].replace('.png', '.txt')
        final_dest = host_output_dir / final_rel_path
        
        # Ensure the subfolder structure matches dataset/images/
        final_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(s2_out_file, final_dest)

print("Pulling global timing summary...")
subprocess.run(f'{ADB} -s {device_id} pull {target_device_dir}/timing_summary.txt {host_output_dir}/', shell=True, check=True)

host_end_time = time.time()
total_duration = host_end_time - host_start_time

print(f"\nTotal Host Wall-Clock Time: {total_duration:.2f} seconds")
print(f"Results saved maintaining directory structure to: {host_output_dir}/")
print(f"Global timing log saved to: {host_output_dir}/timing_summary.txt")

# Cleanup Local Shell Scripts
for script in [script_s1_name, script_s2_name]:
    if os.path.exists(script):
        os.remove(script)

script_end_time = time.time()
total_duration = script_end_time - script_start_time
print(f"\nTotal Script Execution Time: {total_duration:.2f} seconds")
