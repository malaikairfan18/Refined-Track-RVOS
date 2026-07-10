import gradio as gr
import alphaclip
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from utils import *
from resam_modules import refine_and_requery_keyframe, SoftSemanticAlignment
import cv2
import os
import imageio
import numpy as np
from PIL import Image
import torch
import torchvision as tv
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')


def segment_video(video_path, prompt, gpu):

    # GPU setting
    torch.cuda.set_device(int(gpu))
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):

        # load data
        reader = imageio.get_reader(video_path)
        fps = reader.get_meta_data().get('fps', 24) 
        frames = [frame for frame in reader]
        reader.close()

        # initialize EVF-SAM
        tokenizer, evfsam = init_models()

        # initialize Alpha-CLIP
        clip, clip_preprocess = alphaclip.load('ViT-L/14@336px', alpha_vision_ckpt_pth='weights/clip_l14_336_grit_20m_4xe.pth', device='cuda')
        clip_preprocess_mask = transforms.Compose([transforms.Resize((336, 336)), transforms.Normalize(0.5, 0.26)])

        # initialize Cutie
        cutie = get_default_model(config='ytvos_config')
        processor = InferenceCore(cutie, cfg=cutie.cfg)

        # input pre-process
        video_len = len(frames)
        imgs_beit = []
        imgs_sam = []
        imgs_clip = []
        imgs_cutie = []
        for i in range(video_len):
            image_np = frames[i]
            original_size_list = [image_np.shape[:2]]

            # BEiT pre-process
            img_beit = beit3_preprocess(Image.fromarray(image_np), 224)
            imgs_beit.append(img_beit)

            # SAM pre-process
            img_sam, resize_shape = sam_preprocess(image_np)
            imgs_sam.append(img_sam)

            # Alpha-CLIP pre-process
            img_clip = clip_preprocess(Image.fromarray(image_np))
            imgs_clip.append(img_clip)

            # Cutie pre-process
            img_cutie = tv.transforms.ToTensor()(Image.fromarray(image_np))
            imgs_cutie.append(img_cutie)

        # per-frame mask prediction
        ref_masks = []
        ref_scores = []
        ref_num = 10
        for ref_idx in range(ref_num):
            i = int(ref_idx * (video_len - 1) / (ref_num - 1))
            words = tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
            ref_mask, ref_score = evfsam.inference(imgs_sam[i].unsqueeze(0).cuda(), imgs_beit[i].unsqueeze(0).cuda(), words, resize_shape, original_size_list)
            ref_mask = (ref_mask > 0).float()
            ref_masks.append(ref_mask)

            # consider vision-text alignment in addition to segmentation confidence
            w1, w2 = 0.5, 0.5
            clip_text = alphaclip.tokenize([prompt]).cuda()
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
        ssa_controller = SoftSemanticAlignment(queue_size=128, embedding_dim=256).cuda()
        
        # Extract visual embedding for the initial refined mask
        with torch.no_grad():
            alpha = clip_preprocess_mask(refined_ref_mask).cuda()
            initial_emb = clip.visual(imgs_clip[best_i].unsqueeze(0).cuda(), alpha.unsqueeze(0))
            initial_emb = initial_emb / initial_emb.norm(dim=-1, keepdim=True)
            
        # Initialize the FIFO queue with the keyframe anchor embedding
        ssa_controller.queue.copy_(initial_emb.repeat(128, 1))

        # color work
        overlay_color = np.array([0, 255, 0], dtype=np.uint8)

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
            mask_np = mask.detach().cpu().numpy() * 255
            mask_np = mask_np.astype(np.uint8)

            # color work
            colored_mask = np.zeros_like(frames[0], dtype=np.uint8)
            colored_mask[mask_np == 255] = overlay_color
            alpha_val = 0.6
            target_frame = frames[i]
            overlayed_image = cv2.addWeighted(colored_mask, alpha_val, target_frame, 1 - alpha_val, 0)
            frames[i] = overlayed_image
            
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
            mask_np = mask.detach().cpu().numpy() * 255
            mask_np = mask_np.astype(np.uint8)

            # color work
            colored_mask = np.zeros_like(frames[0], dtype=np.uint8)
            colored_mask[mask_np == 255] = overlay_color
            alpha_val = 0.6
            target_frame = frames[i]
            overlayed_image = cv2.addWeighted(colored_mask, alpha_val, target_frame, 1 - alpha_val, 0)
            frames[i] = overlayed_image
        
        # save output
        output_filename = 'sample/result.mp4'
        writer = imageio.get_writer(output_filename, fps=fps, codec='libx264')
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        return 'sample/result.mp4'


# gradio setting
demo = gr.Interface(
    fn=segment_video,
    inputs=[gr.Video(label='Input Video'), gr.Text(label='Text Prompt'), gr.Text(label='GPU Number')],
    outputs=gr.Video(label='Output Mask'),
    title='FindTrack Demo Page',
    examples=[
        ['sample/agility.mp4', 'A dog running on grass.', 0],
        ['sample/elon.mp4', 'Elon Musk dancing in a suit.', 0],
        ['sample/trump.mp4', 'Donald Trump dancing and clapping in front of an audience.', 0]
    ],
    allow_flagging="never"
)


if __name__ == '__main__':
    demo.launch(share=True)
