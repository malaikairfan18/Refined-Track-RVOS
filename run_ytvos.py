import alphaclip
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from utils import *
from resam_modules import refine_and_requery_keyframe, SoftSemanticAlignment
import os
import shutil
import cv2
import json
import numpy as np
from PIL import Image
import torch
import torchvision as tv
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig
import warnings
warnings.filterwarnings('ignore')


def test(args):

    # initialize EVF-SAM
    tokenizer, evfsam = init_models(cache_dir=args.cache_dir)

    # initialize Alpha-CLIP
    clip, clip_preprocess = alphaclip.load('ViT-L/14@336px', alpha_vision_ckpt_pth=args.clip_weights, device='cuda')
    clip_preprocess_mask = transforms.Compose([transforms.Resize((336, 336)), transforms.Normalize(0.5, 0.26)])

    # initialize Cutie
    cutie = get_default_model(config='ytvos_config')
    processor = InferenceCore(cutie, cfg=cutie.cfg)

    # load videos
    output_dir = args.output_dir
    save_path_prefix = os.path.join(output_dir, 'Ref_YTVOS_val')
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    
    root_path = args.dataset_path
    img_folder = None
    meta_file = None
    test_meta_file = None

    print(f"Scanning dataset path for files: {root_path}")
    for dirpath, dirnames, filenames in os.walk(root_path):
        if 'JPEGImages' in dirnames and 'valid' in dirpath.lower() and not img_folder:
            img_folder = os.path.join(dirpath, 'JPEGImages')
        if 'meta_expressions.json' in filenames:
            if 'valid' in dirpath.lower() and not meta_file:
                meta_file = os.path.join(dirpath, 'meta_expressions.json')
            elif 'test' in dirpath.lower() and not test_meta_file:
                test_meta_file = os.path.join(dirpath, 'meta_expressions.json')

    print(f"🔍 Auto-detected image folder: {img_folder}")
    print(f"🔍 Auto-detected validation metadata: {meta_file}")
    print(f"🔍 Auto-detected test metadata: {test_meta_file}")

    if not img_folder or not meta_file or not test_meta_file:
        raise FileNotFoundError("❌ Could not auto-detect JPEGImages or meta_expressions.json inside the provided dataset path.")

    with open(meta_file, 'r') as f:
        data = json.load(f)['videos']
    valid_test_videos = set(data.keys())
    with open(test_meta_file, 'r') as f:
        test_data = json.load(f)['videos']
    test_videos = set(test_data.keys())
    valid_videos = valid_test_videos - test_videos
    video_list = sorted([video for video in valid_videos])

    # inference
    for idx_, video in enumerate(video_list):
        print(idx_)
        metas = []
        expressions = data[video]['expressions']
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        for i in range(num_expressions):
            meta = {}
            meta['video'] = video
            meta['exp'] = expressions[expression_list[i]]['exp']
            meta['exp_id'] = expression_list[i]
            meta['frames'] = data[video]['frames']
            metas.append(meta)
        meta = metas
        video_name = video
        frames = data[video]['frames']
        video_len = len(frames)

        # input pre-process
        imgs_beit = []
        imgs_sam = []
        imgs_clip = []
        imgs_cutie = []
        for i in range(video_len):
            img_path = os.path.join(img_folder, video_name, frames[i] + '.jpg')
            image_np = cv2.imread(img_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            original_size_list = [image_np.shape[:2]]

            # BEiT pre-process
            img_beit = beit3_preprocess(Image.open(img_path), 224)
            imgs_beit.append(img_beit)

            # SAM pre-process
            img_sam, resize_shape = sam_preprocess(image_np)
            imgs_sam.append(img_sam)

            # Alpha-CLIP pre-process
            img_clip = clip_preprocess(Image.open(img_path))
            imgs_clip.append(img_clip)

            # Cutie pre-process
            img_cutie = tv.transforms.ToTensor()(Image.open(img_path))
            imgs_cutie.append(img_cutie)

        # for each language
        for e in range(num_expressions):

            # make files
            video_name = meta[e]['video']
            exp = meta[e]['exp']
            exp_id = meta[e]['exp_id']
            frames = meta[e]['frames']
            save_path = os.path.join(save_path_prefix, video_name, exp_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            elif len(os.listdir(save_path)) == len(frames):
                print(f"Skipping {video_name} - {exp_id}, already completely processed.")
                continue

            # per-frame mask prediction
            ref_masks = []
            ref_scores = []
            ref_num = 10
            for ref_idx in range(ref_num):
                i = int(ref_idx * (video_len - 1) / (ref_num - 1))
                words = tokenizer(exp, return_tensors='pt')['input_ids'].cuda()
                ref_mask, ref_score = evfsam.inference(imgs_sam[i].unsqueeze(0).cuda(), imgs_beit[i].unsqueeze(0).cuda(), words, resize_shape, original_size_list)
                ref_mask = (ref_mask > 0).float()
                ref_masks.append(ref_mask)

                # consider vision-text alignment in addition to segmentation confidence
                w1, w2 = 0.5, 0.5
                clip_text = alphaclip.tokenize([exp]).cuda()
                alpha = clip_preprocess_mask(ref_mask).cuda()
                image_features = clip.visual(imgs_clip[i].unsqueeze(0).cuda(), alpha.unsqueeze(0))
                text_features = clip.encode_text(clip_text)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                ref_score = w1 * ref_score + w2 * torch.matmul(image_features, text_features.transpose(0, 1))[0]
                ref_scores.append(ref_score)

            # select reference frame with highest mask score
            best_ref_idx = torch.argmax(torch.stack(ref_scores, dim=0), dim=0)
            best_i = int(best_ref_idx * (video_len - 1) / (ref_num - 1))

            # Apply R³-Loop Keyframe Refinement & SAM Box Re-query
            refined_ref_mask = refine_and_requery_keyframe(
                ref_masks[best_ref_idx], 
                imgs_sam[best_i], 
                evfsam, 
                resize_shape, 
                original_size_list
            )

            # Initialize SSA Module
            ssa_controller = SoftSemanticAlignment(queue_size=128, embedding_dim=768).cuda()
            
            # Extract visual embedding for the initial refined mask
            with torch.no_grad():
                alpha = clip_preprocess_mask(refined_ref_mask).cuda()
                initial_emb = clip.visual(imgs_clip[best_i].unsqueeze(0).cuda(), alpha.unsqueeze(0))
                initial_emb = initial_emb / initial_emb.norm(dim=-1, keepdim=True)
                
            # Initialize the FIFO queue with the keyframe anchor embedding
            ssa_controller.queue.copy_(initial_emb.repeat(128, 1))

            # forward pass
            for i in range(best_i, video_len):
                if i == best_i:
                    mask_prob = processor.step(imgs_cutie[i].cuda(), refined_ref_mask.squeeze(0), objects=[1])
                else:
                    mask_prob = processor.step(imgs_cutie[i].cuda())
                mask = processor.output_prob_to_mask(mask_prob).float()

                # SSA Gated Memory Update check
                with torch.no_grad():
                    alpha = clip_preprocess_mask(mask.unsqueeze(0)).cuda()
                    current_emb = clip.visual(imgs_clip[i].unsqueeze(0).cuda(), alpha.unsqueeze(0))
                    current_emb = current_emb / current_emb.norm(dim=-1, keepdim=True)
                    
                    loss_ssa = ssa_controller(current_emb, update_queue=False)
                    similarity = 1.0 - loss_ssa.item()
                    
                # If similarity is low, we gate memory update to prevent poisoning
                if similarity < 0.6:
                    processor.mem_every = 999999
                else:
                    processor.mem_every = processor.cfg.mem_every
                    # Enqueue only stable embeddings to the queue
                    ssa_controller._dequeue_and_enqueue(current_emb)

                # clear memory for each sequence
                if i == video_len - 1:
                    processor.clear_memory()

                # convert format
                mask_np = mask.detach().cpu().numpy().astype(np.float32)
                mask_img = Image.fromarray(mask_np * 255).convert('L')
                save_file = os.path.join(save_path, frames[i] + '.png')
                mask_img.save(save_file)

            # Re-initialize SSA queue for backward pass to default to anchor state
            ssa_controller.queue.copy_(initial_emb.repeat(128, 1))
            ssa_controller.ptr.zero_()
            processor.mem_every = processor.cfg.mem_every

            # backward pass
            for i in range(best_i, -1, -1):
                if i == best_i:
                    mask_prob = processor.step(imgs_cutie[i].cuda(), refined_ref_mask.squeeze(0), objects=[1])
                else:
                    mask_prob = processor.step(imgs_cutie[i].cuda())
                mask = processor.output_prob_to_mask(mask_prob).float()

                # SSA Gated Memory Update check
                with torch.no_grad():
                    alpha = clip_preprocess_mask(mask.unsqueeze(0)).cuda()
                    current_emb = clip.visual(imgs_clip[i].unsqueeze(0).cuda(), alpha.unsqueeze(0))
                    current_emb = current_emb / current_emb.norm(dim=-1, keepdim=True)
                    
                    loss_ssa = ssa_controller(current_emb, update_queue=False)
                    similarity = 1.0 - loss_ssa.item()
                    
                if similarity < 0.6:
                    processor.mem_every = 999999
                else:
                    processor.mem_every = processor.cfg.mem_every
                    ssa_controller._dequeue_and_enqueue(current_emb)

                # clear memory for each sequence
                if i == 0:
                    processor.clear_memory()

                # convert format
                mask_np = mask.detach().cpu().numpy().astype(np.float32)
                mask_img = Image.fromarray(mask_np * 255).convert('L')
                save_file = os.path.join(save_path, frames[i] + '.png')
                mask_img.save(save_file)

    print(f"Zipping results for Codabench submission...")
    zip_name = shutil.make_archive(save_path_prefix, 'zip', save_path_prefix)
    print(f"Created zip file: {zip_name}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Refined Track RVOS - YouTube-VOS inference")
    parser.add_argument('--dataset_path', type=str, default='../DB/RVOS/YTVOS', help='Path to YTVOS dataset')
    parser.add_argument('--cache_dir', type=str, default='../huggingface', help='HuggingFace cache directory')
    parser.add_argument('--clip_weights', type=str, default='weights/clip_l14_336_grit_20m_4xe.pth', help='Path to Alpha-CLIP weights')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()

    torch.cuda.set_device(0)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        test(args)
