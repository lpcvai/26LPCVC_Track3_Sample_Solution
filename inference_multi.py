import os
import argparse
import json
import glob

import numpy as np

import torch

import qai_hub as hub


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_model', default='Snapdragon 8 Elite QRD', help='Device model.', required=False, type=str)
    parser.add_argument('--model_id', default='', help='Device model.', required=False, type=str)
    parser.add_argument('--load_path', default='./example1/Example1B/outputdir/inputs*.pt', help='Path to load input data.', required=False, type=str)
    parser.add_argument('--out_path', default='./in_mul_out/', help='Path to save output.', required=False, type=str)
    parser.add_argument('--batch', default=0, help='Batch data to submit.', required=False, type=int)
    args = parser.parse_args()


    device_model = args.device_model

    ### optimize the onnx file ###
    model = "./weight_sharing_model_1_of_1.serialized.bin" ### first submission needs local model
    # model_id = args.model_id ### later submissions ok with model id
    # model = hub.get_model(args.model_id)

    graph_name = "ar1_cl2048_1_of_1"


    load_input_path = args.load_path

    input_paths = glob.glob(load_input_path)
    input_paths.sort()

    inputs_sample = {}
    inputs_sample["attention_mask"] = []
    inputs_sample['inputs_embeds'] = []
    inputs_sample['position_ids_cos'] = []
    inputs_sample['position_ids_sin'] = []

    for _i in range(0, 28):
        inputs_sample[f'past_key_{_i}_in'] = []
        inputs_sample[f'past_value_{_i}_in'] = []


    cnt = 0
    for input_path in input_paths:

        if cnt >= 10*args.batch and cnt < 10*(args.batch+1):

            input_data = torch.load(input_path)

            # print("attention_mask", input_data['attention_mask'].shape)
            # print("inputs_embeds", input_data['inputs_embeds'].shape)
            # print("position_ids", input_data['position_ids'][0].shape, input_data['position_ids'][1].shape)
            # print("past_key_values", input_data['past_key_values'][0][0].shape, input_data['past_key_values'][0][1].shape)
            # breakpoint()

            inputs_sample["attention_mask"].append(input_data['attention_mask'].cpu().numpy().astype(np.float32))
            inputs_sample['inputs_embeds'].append(input_data['inputs_embeds'].cpu().numpy().astype(np.float32))
            inputs_sample['position_ids_cos'].append(input_data['position_ids'][0].cpu().numpy().astype(np.float32))
            inputs_sample['position_ids_sin'].append(input_data['position_ids'][1].cpu().numpy().astype(np.float32))

            for _i in range(0, 28):
                inputs_sample[f'past_key_{_i}_in'].append(np.transpose(input_data['past_key_values'][_i][0].cpu().numpy().astype(np.float32), (1,0,2,3)))
                inputs_sample[f'past_value_{_i}_in'].append(np.transpose(input_data['past_key_values'][_i][1].cpu().numpy().astype(np.float32), (1,0,2,3)))

        cnt += 1


    print("start uploading data")
    inference_job = hub.submit_inference_job(
        model=model,
        device=hub.Device(device_model),
        options="--qnn_options context_enable_graphs=ar1_cl2048_1_of_1",
        inputs=inputs_sample
    )
    output = inference_job.download_output_data()

    for k, v in output.items():
        path = args.out_path + f"/submission_{args.batch}/output_{k}.npy"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, v)


    print("processing completed!")
