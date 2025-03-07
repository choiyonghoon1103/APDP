import argparse
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
import os
from prompt_to_prompt.ptp_classes import AttentionStore, AttentionReplace, AttentionRefine, EmptyControl,load_512
from prompt_to_prompt.ptp_utils import register_attention_control, text2image_ldm_stable, view_images
from ddm_inversion.inversion_utils import  inversion_forward_process, inversion_reverse_process
from ddm_inversion.utils import image_grid,dataset_from_yaml

from torch import autocast, inference_mode
from ddm_inversion.ddim_inversion import ddim_inversion

from llava_simple import mask_similar_words, aesthetic_score, llava_output
import calendar
import time
from clip_score import clip_sim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--cfg_src", type=float, default=3.5)
    parser.add_argument("--cfg_tar", type=float, default=15)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--eta", type=float, default=1)
    parser.add_argument("--mode",  default="our_inv", help="modes: our_inv,p2pinv,p2pddim,ddim")
    parser.add_argument("--skip",  type=int, default=36)
    parser.add_argument("--xa", type=float, default=0.6)
    parser.add_argument("--sa", type=float, default=0.2)
    parser.add_argument("--image_path", type=str, default="./chair.jpeg")
    parser.add_argument("--source_cls", "-sc",type=str, default="")
    parser.add_argument("--target_cls", "-tc",type=str, default="")
    args = parser.parse_args()

    model_id = "CompVis/stable-diffusion-v1-4"

    device = f"cuda:{args.device_num}"

    cfg_scale_src = args.cfg_src
    cfg_scale_tar_list = [args.cfg_tar]
    eta = args.eta 
    skip_zs = [args.skip]
    xa_sa_string = f'_xa_{args.xa}_sa{args.sa}_' if args.mode=='p2pinv' else '_'

    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)

    ldm_stable = StableDiffusionPipeline.from_pretrained(model_id).to(device)


    image_path = args.image_path
    image_folder = image_path.split('/')[1] 
    prompt_src = mask_similar_words(llava_output(args.image_path), args.source_cls, args.source_cls, similarity_threshold = 0.5)
    prompt_tar_list = [mask_similar_words(prompt_src, args.source_cls, args.target_cls)]

    if args.mode=="p2pddim" or args.mode=="ddim":
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        ldm_stable.scheduler = scheduler
    else:
        ldm_stable.scheduler = DDIMScheduler.from_config(model_id, subfolder = "scheduler")
        
    ldm_stable.scheduler.set_timesteps(args.num_diffusion_steps)

    offsets=(0,0,0,0)
    x0 = load_512(image_path, *offsets, device)

    with autocast("cuda"), inference_mode():
        w0 = (ldm_stable.vae.encode(x0).latent_dist.mode() * 0.18215).float()

    if args.mode=="p2pddim" or args.mode=="ddim":
        wT = ddim_inversion(ldm_stable, w0, prompt_src, cfg_scale_src)
    else:
        wt, zs, wts = inversion_forward_process(ldm_stable, w0, etas=eta, prompt=prompt_src, cfg_scale=cfg_scale_src, prog_bar=True, num_inference_steps=args.num_diffusion_steps)

    for k in range(len(prompt_tar_list)):
        prompt_tar = prompt_tar_list[k]
        save_path = os.path.join(f'./results/', args.mode+xa_sa_string+str(time_stamp), image_path.split(sep='.')[0], 'src_' + args.source_cls, 'dec_' + args.target_cls)
        os.makedirs(save_path, exist_ok=True)

        src_tar_len_eq = (len(prompt_src.split(" ")) == len(prompt_tar.split(" ")))

        for cfg_scale_tar in cfg_scale_tar_list:
            for skip in skip_zs:    
                if args.mode=="our_inv":
                    controller = AttentionStore()
                    register_attention_control(ldm_stable, controller)
                    w0, _ = inversion_reverse_process(ldm_stable, xT=wts[args.num_diffusion_steps-skip], etas=eta, prompts=[prompt_tar], cfg_scales=[cfg_scale_tar], prog_bar=True, zs=zs[:(args.num_diffusion_steps-skip)], controller=controller)

                elif args.mode=="p2pinv":
                    cfg_scale_list = [cfg_scale_src, cfg_scale_tar]
                    prompts = [prompt_src, prompt_tar]
                    if src_tar_len_eq:
                        controller = AttentionReplace(prompts, args.num_diffusion_steps, cross_replace_steps=args.xa, self_replace_steps=args.sa, model=ldm_stable)
                    else:
                        controller = AttentionRefine(prompts, args.num_diffusion_steps, cross_replace_steps=args.xa, self_replace_steps=args.sa, model=ldm_stable)

                    register_attention_control(ldm_stable, controller)
                    w0, _ = inversion_reverse_process(ldm_stable, xT=wts[args.num_diffusion_steps-skip], etas=eta, prompts=prompts, cfg_scales=cfg_scale_list, prog_bar=True, zs=zs[:(args.num_diffusion_steps-skip)], controller=controller)
                    w0 = w0[1].unsqueeze(0)

                elif args.mode=="p2pddim" or args.mode=="ddim":
                    if skip != 0:
                        continue
                    prompts = [prompt_src, prompt_tar]
                    if args.mode=="p2pddim":
                        if src_tar_len_eq:
                            controller = AttentionReplace(prompts, args.num_diffusion_steps, cross_replace_steps=.8, self_replace_steps=0.4, model=ldm_stable)
                        else:
                            controller = AttentionRefine(prompts, args.num_diffusion_steps, cross_replace_steps=.8, self_replace_steps=0.4, model=ldm_stable)
                    else:
                        controller = EmptyControl()

                    register_attention_control(ldm_stable, controller)
                    cfg_scale_list = [cfg_scale_src, cfg_scale_tar]
                    w0, latent = text2image_ldm_stable(ldm_stable, prompts, controller, args.num_diffusion_steps, cfg_scale_list, None, wT)
                    w0 = w0[1:2]
                else:
                    raise NotImplementedError
                
                with autocast("cuda"), inference_mode():
                    x0_dec = ldm_stable.vae.decode(1 / 0.18215 * w0).sample
                if x0_dec.dim()<4:
                    x0_dec = x0_dec[None,:,:,:]
                img = image_grid(x0_dec)

                current_GMT = time.gmtime()
                time_stamp_name = calendar.timegm(current_GMT)
                image_name_png = f'cfg_d_{cfg_scale_tar}_' + f'skip_{skip}_{time_stamp_name}' + ".png"

                save_full_path = os.path.join(save_path, image_name_png)
                s_sim_score = clip_sim(img, prompt_tar_list[0])
                t_sim_score = clip_sim(img, prompt_src)
                img_aesthetic_score = aesthetic_score(img)

                print(f"Source Prompt: {prompt_src}")
                print(f"Target Prompt: {prompt_tar_list[0]}")
                print(f"Generation image - target prompt clip score: {s_sim_score}")
                print(f"Generation image - source prompt clip score: {t_sim_score}")
                print(f"Generation image Aesthetic_score: {img_aesthetic_score}")
                img.save(save_full_path)