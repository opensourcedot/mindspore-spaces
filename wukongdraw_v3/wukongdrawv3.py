# reference to https://github.com/Stability-AI/generative-models
import argparse
import ast
import time
import pickle

from gm.helpers import get_batch, get_unique_embedder_keys_from_conditioner
from gm.helpers import WuKong_RATIOS, VERSION2SPECS, create_model, init_sampling
from gm.util import seed_everything
from omegaconf import OmegaConf

import redis
import numpy as np
from PIL import Image
import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net


def get_parser_sample():
    parser = argparse.ArgumentParser(description="sampling with sd-xl")
    parser.add_argument("--task", type=str, default="txt2img", choices=["txt2img", "img2img"])
    parser.add_argument("--config", type=str, default="configs/inference/sd_xl_base_wukong.yaml")
    parser.add_argument("--weight", type=str, default="checkpoints/pangu_low_timestamp-127da122.ckpt")
    parser.add_argument("--high_timestamp_weight", type=str, default="checkpoints/pangu_high_timestamp-c6344411.ckpt")
    parser.add_argument("--use_high_timestamp_model", action="store_true")
    parser.add_argument("--queue_name", type=str, default="request_queue_v3_for_image")
    parser.add_argument("--pre_compile", action="store_true")
    parser.add_argument("--sample_step", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampler", type=str, default="EulerEDMSampler")
    parser.add_argument("--guider", type=str, default="VanillaCFG")
    parser.add_argument("--discretization", type=str, default="LegacyDDPMDiscretization")
    parser.add_argument("--device_target", type=str, default="Ascend", help="device target, Ascend/GPU/CPU")
    parser.add_argument(
        "--ms_mode", type=int, default=1, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=1)"
    )
    parser.add_argument("--ms_jit", type=ast.literal_eval, default=True, help="use jit or not")
    parser.add_argument("--ms_amp_level", type=str, default="O2")
    parser.add_argument(
        "--ms_enable_graph_kernel", type=ast.literal_eval, default=False, help="use enable_graph_kernel or not"
    )


    return parser


def run_txt2img(
    args,
    model,
    version_dict
):

    negative_prompt_local = "重复，重复物体，重复花纹，repeat"
    queue_name = args.queue_name
    num_samples = [1]
    redis_client = redis.Redis(host="0.0.0.0", port=6379, db=0)
    pre_compile = args.pre_compile
    pre_compile_resolution_list = ["1:4", "4:15",
    "1:2",
    "9:16",
    "3:4",
    "1:1",
    "4:3",
    "16:9",
    "2:1",
    "15:4",
    "4:1"]

    model_weight = load_checkpoint(args.weight)
    if args.use_high_timestamp_model and args.high_timestamp_weight:
        high_timestamp_weight = load_checkpoint(args.high_timestamp_weight)
    else:
        high_timestamp_weight = None

    load_param_into_net(model, model_weight)

    while True:
        if pre_compile and pre_compile_resolution_list:
            base_count = 0
            request_token = "pre_compile"
            input_txt = "a cute cat"
            enhanced_txt = "a cute cat"
            negative_prompt = negative_prompt_local
            aspect_ratio_txt = pre_compile_resolution_list.pop()
            guidance_scale = 6.0
            aesthetic_scale = 4.0
            anime_scale = 0.0
            photo_scale = 0.0
            version_name = "v3"
        else:
            request = redis_client.lpop(queue_name)
            if request is None or len(request) == 0:
                time.sleep(1)
                continue
            info = pickle.loads(request)
            base_count = info[0]
            request_token = info[1]
            input_txt = info[2]
            enhanced_txt = info[3]
            negative_prompt = info[4]
            aspect_ratio_txt = info[5]
            guidance_scale = info[6]
            aesthetic_scale = info[7]
            anime_scale = info[8]
            photo_scale = info[9]
            version_name = info[-1]

        target_size_as_ind, W, H = WuKong_RATIOS[aspect_ratio_txt]
        C = version_dict["C"]
        F = version_dict["f"]

        print("Txt2Img Sampling")
        s_time = time.time()
        print("Sampling")


    assert args.sd_xl_base_ratios in SD_XL_BASE_RATIOS
    W, H = SD_XL_BASE_RATIOS[args.sd_xl_base_ratios]
    C = version_dict["C"]
    F = version_dict["f"]
    value_dict = {
        "prompt": enhanced_txt,
        "negative_prompt": negative_prompt,
        "aesthetic_scale": aesthetic_scale,
        "anime_scale": anime_scale,
        "photo_scale": photo_scale,
        "target_size_as_ind": target_size_as_ind,
    }

    batch, batch_uc, other_batch, other_scale = get_batch(
        get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, num_samples, dtype=ms.float16
    )
    sampler, _, _ = init_sampling(
        sampler=args.sampler,
        num_cols=1,
        guider=args.guider,
        guidance_scale=guidance_scale,
        discretization=args.discretization,
        steps=args.sample_step,
        stage2strength=None,
        other_scale=other_scale,
    )
    for key in batch:
        if isinstance(batch[key], Tensor):
            print(key, batch[key].shape)
        elif isinstance(batch[key], list):
            print(key, [len(i) for i in batch[key]])
        else:
            print(key, batch[key])
    print("Get Condition Done.")

    print("Embedding Starting...")
    c, uc, other_c = model.conditioner.get_unconditional_conditioning(
        batch,
        batch_uc=batch_uc,
        force_uc_zero_embeddings=None,
        other_batch=other_batch
    )
    print("Embedding Done.")
    for k in c:
        if not k == "crossattn":
            c[k], uc[k] = map(
                lambda y :y[k][:int(np.prod(num_samples))],
                (c, uc)
            )
        for _c in other_c:
            _c[k] = _c[k][:int(np.prod(num_samples))]


    shape = (np.prod(num_samples), C, H // F, W // F)
    randn = Tensor(np.random.randn(*shape), ms.float32)

    print("Sampling latent Starting....")
    samples_z = sampler(model, model_weight, high_timestamp_weight, randn, cond=c, uc=uc, other_c=other_c)
    print("Sampling latent Done")

    print("Decode latent Starting....")
    samples_x = model.decode_first_stage(samples_z)
    samples_x = samples_x.asnumpy()
    print("Decode latent Done")

    samples = np.clip((samples_x + 1.0) / 2.0, a_min=0.0, a_max=1.0)
    for sample in samples:
        sample = 255.0 * sample.transpose(1, 2, 0)
        img = Image.fromarray(sample.astype(np.uint8))
        img_name = f"{input_txt[:50]}-{request_token}-{base_count:05}.png"
        redis_client.lpush(request_token, pickle.dumps((img_name, img)))
    print(f"Txt2Img sample step {smaller.num_steps}, time cost:{time.time() - s_time:.2f}s")

    return


def sample():
    parser = get_parser_sample()
    args, _ = parser.parse_known_args()
    ms.context.set_context(mode=args.ms_mode, device_target=args.device_target)

    config = OmegaConf.load(args.config)
    version = config.pop("version", "SDXL-base-1.0")
    version_dict = VERSION2SPECS.get(version)

    task = args.task
    seed_everything(args.seed)

    # Init Model
    model, filter = create_model(
        config,
        checkpoints=None,
        freeze=True,
        load_filter=False,
        amp_level=args.ms_amp_level
    )  # TODO: Add filter support

    if task == "txt2img":
        run_txt2img(
            args,
            model,
            version_dict
        )
    else:
        raise ValueError(f"Unknown task {task}")

if __name__ == "__main__":
    examples = [
        ['prompts = "一只可爱的猫."'],
        ['prompts = "幽兰深谷 写实."']
]

    demo = gr.Interface(
        fn=sample,
        inputs=[
            gr.Textbox(
                label="prompts",
                placeholder="输入文本",
                lines=2,
            ),
        ],
        outputs=[
            gr.Image(
                label="生成图片结果",
                type="pil"  # 指定返回的是 PIL 图像对象
            )
        ],
        submit_btn=gr.Button("生成图片", variant="primary"),
        title="wukong draw v3",
        description="wukong draw v3 is a txt2img generation model based on MindSpore.",
        theme="soft",
        examples=examples,
        cache_examples=False,
        allow_flagging="never",
    )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

