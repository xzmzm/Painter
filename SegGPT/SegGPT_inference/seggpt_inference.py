import os
import argparse

import torch
import numpy as np
import torch.onnx
import cv2
import onnxruntime as ort

from seggpt_engine import inference_image, inference_video
import models_seggpt


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def get_args_parser():
    parser = argparse.ArgumentParser('SegGPT inference', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt',
                        default='seggpt_vit_large.pth')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='seggpt_vit_large_patch16_input896x448')
    parser.add_argument('--input_image', type=str, help='path to input image to be tested',
                        default=None)
    parser.add_argument('--input_video', type=str, help='path to input video to be tested',
                        default=None)
    parser.add_argument('--num_frames', type=int, help='number of prompt frames in video',
                        default=0)
    parser.add_argument('--prompt_image', type=str, nargs='+', help='path to prompt image',
                        default=None)
    parser.add_argument('--prompt_target', type=str, nargs='+', help='path to prompt target',
                        default=None)
    parser.add_argument('--seg_type', type=str, help='embedding for segmentation types', 
                        choices=['instance', 'semantic'], default='instance')
    parser.add_argument('--device', type=str, help='cuda or cpu',
                        default='cuda')
    parser.add_argument('--output_dir', type=str, help='path to output',
                        default='./')
    parser.add_argument('--export_onnx', action='store_true', help='Export model to ONNX format')
    parser.add_argument('--onnx_path', type=str, default='seggpt_model.onnx', help='Path to save ONNX model')
    parser.add_argument('--fp16', action='store_true', help='Export model in FP16 format')
    parser.add_argument('--use_onnx', action='store_true', help='Use ONNX model for inference')
    return parser.parse_args()


def prepare_model(chkpt_dir: str, arch: str = 'seggpt_vit_large_patch16_input896x448', seg_type: str = 'instance') -> torch.nn.Module:
    model = getattr(models_seggpt, arch)()
    model.seg_type = seg_type
    try:
        checkpoint = torch.load(chkpt_dir, weights_only=True)
        msg = model.load_state_dict(checkpoint['model'], strict=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {chkpt_dir}: {str(e)}")
    model.eval()
    return model


class SegGPTONNX(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, imgs, tgts, bool_masked_pos, seg_type):
        latent = self.model.forward_encoder(imgs, tgts, bool_masked_pos, seg_type)
        pred = self.model.forward_decoder(latent)
        pred = self.model.patchify(pred)
        pred = self.model.unpatchify(pred)
        pred = torch.einsum('nchw->nhwc', pred)
        pred = pred[0, pred.shape[1]//2:, :, :]
        return pred


def export_to_onnx(model, save_path, input_shape=(1, 896, 448, 3), device='cuda', use_fp16=False):
    """
    Export SegGPT model to ONNX format with same preprocessing as inference
    """
    # Wrap the model for ONNX export
    onnx_model = SegGPTONNX(model)
    onnx_model.eval()
    
    if use_fp16:
        onnx_model.half()
        
    # Match preprocessing from inference_image and run_one_image
    batch_size, height, width, channels = input_shape
    
    # Create dummy inputs in NHWC format first, explicitly as float32
    dummy_img = torch.randn(batch_size, height, width, channels, dtype=torch.float32, device=device)
    dummy_tgt = torch.randn(batch_size, height, width, channels, dtype=torch.float32, device=device)
    
    # Normalize by ImageNet mean and std (same as in inference_image)
    dummy_img = (dummy_img - torch.tensor(imagenet_mean, dtype=torch.float32, device=device)) / torch.tensor(imagenet_std, dtype=torch.float32, device=device)
    dummy_tgt = (dummy_tgt - torch.tensor(imagenet_mean, dtype=torch.float32, device=device)) / torch.tensor(imagenet_std, dtype=torch.float32, device=device)
    
    # Convert to NCHW format (same as in run_one_image)
    dummy_img = torch.einsum('nhwc->nchw', dummy_img)
    dummy_tgt = torch.einsum('nhwc->nchw', dummy_tgt)
    
    # Create other inputs matching run_one_image
    dummy_mask = torch.zeros(model.patch_embed.num_patches, dtype=torch.float32, device=device)
    dummy_mask[model.patch_embed.num_patches//2:] = 1
    dummy_mask = dummy_mask.unsqueeze(dim=0)
    
    # Set seg_type based on model configuration
    if model.seg_type == 'instance':
        dummy_seg_type = torch.ones((batch_size, 1), dtype=torch.float32, device=device)
    else:
        dummy_seg_type = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)

    dynamic_axes = {
        'input_image': {0: 'batch_size'},
        'input_target': {0: 'batch_size'},
        'input_mask': {0: 'batch_size'},
        'input_seg_type': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }

    try:
        if use_fp16:
            dummy_img = dummy_img.half()
            dummy_tgt = dummy_tgt.half()
            
        torch.onnx.export(
            onnx_model,
            (dummy_img, dummy_tgt, dummy_mask, dummy_seg_type),  # Removed valid input
            save_path,
            input_names=['imgs', 'tgts', 'bool_masked_pos', 'seg_type'],  # Updated input names
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            opset_version=15,
            do_constant_folding=True,
            verbose=False
        )
        print(f"Model successfully exported to: {save_path}")
        
        if use_fp16:
            onnx_model.float()
            
    except Exception as e:
        if use_fp16:
            onnx_model.float()
        print(f"Error exporting model to ONNX: {str(e)}")


def get_available_providers():
    """Get available ONNX Runtime providers"""
    providers = []
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.append('CUDAExecutionProvider')
    if 'DmlExecutionProvider' in ort.get_available_providers():
        providers.append('DmlExecutionProvider')
    providers.append('CPUExecutionProvider')
    return providers


def inference_image_onnx(onnx_path: str, device: str, input_image: str, prompt_image: list, prompt_target: list, output_path: str):
    """Run inference using ONNX model"""
    providers = get_available_providers()
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Load and preprocess images
    img = cv2.imread(input_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = img.shape[:2]
    input_image_original = img.copy()  # Store original for blending
    
    # Resize to model input size
    img = cv2.resize(img, (448, 896))
    
    prompt_img = cv2.imread(prompt_image[0])
    prompt_img = cv2.cvtColor(prompt_img, cv2.COLOR_BGR2RGB)
    prompt_img = cv2.resize(prompt_img, (448, 896))
    
    # Normalize images exactly as in PyTorch mode
    img = img / 255.0
    prompt_img = prompt_img / 255.0
    
    # Apply ImageNet normalization
    img = (img - imagenet_mean) / imagenet_std
    prompt_img = (prompt_img - imagenet_mean) / imagenet_std
    
    # Convert to NCHW format (same as PyTorch)
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
    prompt_img = torch.from_numpy(prompt_img).float().permute(2, 0, 1).unsqueeze(0)
    
    # Create mask (same as PyTorch)
    num_patches = (896 // 16) * (448 // 16)
    dummy_mask = torch.zeros(num_patches, dtype=torch.float32)
    dummy_mask[num_patches//2:] = 1
    dummy_mask = dummy_mask.unsqueeze(dim=0)
    
    # Set seg_type for instance segmentation
    seg_type = torch.ones((1, 1), dtype=torch.float32)
    
    # Run inference
    ort_inputs = {
        'imgs': img.numpy(),
        'tgts': prompt_img.numpy(),
        'bool_masked_pos': dummy_mask.numpy(),
        'seg_type': seg_type.numpy()
    }
    
    pred = session.run(None, ort_inputs)[0]
    
    # Post-process output (already in correct format from ONNX model)
    pred = torch.from_numpy(pred)
    
    # Denormalize and scale
    pred = torch.clip((pred * imagenet_std + imagenet_mean) * 255, 0, 255)
    pred = pred.numpy().astype(np.uint8)
    
    # Resize and blend
    pred = cv2.resize(pred, (original_size[1], original_size[0]))
    output = input_image_original * (0.6 * pred / 255 + 0.4)
    output = output.astype(np.uint8)
    
    # Save output
    cv2.imwrite(output_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    print(f'Results saved to {output_path}')


if __name__ == '__main__':
    args = get_args_parser()

    if args.export_onnx:
        device = torch.device(args.device)
        model = prepare_model(args.ckpt_path, args.model, args.seg_type).to(device)
        print('Model loaded.')
        
        export_to_onnx(
            model, 
            args.onnx_path,
            input_shape=(1, 896, 448, 3),
            device=args.device,
            use_fp16=args.fp16
        )
        print('ONNX export completed.')
        exit(0)

    if not (args.input_image or args.input_video) or (args.input_image and args.input_video):
        raise ValueError("Must provide either input_image OR input_video, but not both")

    if args.input_image is not None:
        if not (args.prompt_image and args.prompt_target):
            raise ValueError("Both prompt_image and prompt_target are required for image inference")

        img_name = os.path.basename(args.input_image)
        out_path = os.path.join(args.output_dir, "output_" + '.'.join(img_name.split('.')[:-1]) + '.png')

        if args.use_onnx:
            inference_image_onnx(args.onnx_path, args.device, args.input_image, args.prompt_image, args.prompt_target, out_path)
        else:
            device = torch.device(args.device)
            model = prepare_model(args.ckpt_path, args.model, args.seg_type).to(device)
            print('Model loaded.')
            inference_image(model, device, args.input_image, args.prompt_image, args.prompt_target, out_path)
    
    if args.input_video is not None:
        assert args.prompt_target is not None and len(args.prompt_target) == 1
        vid_name = os.path.basename(args.input_video)
        out_path = os.path.join(args.output_dir, "output_" + '.'.join(vid_name.split('.')[:-1]) + '.mp4')

        inference_video(model, device, args.input_video, args.num_frames, args.prompt_image, args.prompt_target, out_path)

    print('Finished.')
