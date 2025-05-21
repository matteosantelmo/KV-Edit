import os
import re
import time
from dataclasses import dataclass
from glob import iglob
import argparse
from einops import rearrange
from PIL import ExifTags, Image
import torch
# import gradio as gr # Removed Gradio
import numpy as np
from flux.sampling import prepare
from flux.util import (configs, load_ae, load_clip, load_t5)
from models.kv_edit import Flux_kv_edit # Make sure this path is correct for your project structure

@dataclass
class SamplingOptions:
    source_prompt: str = ''
    target_prompt: str = ''
    width: int = 1366
    height: int = 768
    inversion_num_steps: int = 0
    denoise_num_steps: int = 0
    skip_step: int = 0
    inversion_guidance: float = 1.0
    denoise_guidance: float = 1.0
    seed: int = 42
    re_init: bool = False
    attn_mask: bool = False
    attn_scale: float = 1.0

def resize_image(image_array, max_width=1360, max_height=768):
    # (Function remains unchanged, but might not be explicitly called below)
    # ... (keep the original resize_image function code here) ...
    if image_array.shape[-1] == 4:
        mode = 'RGBA'
    else:
        mode = 'RGB'

    pil_image = Image.fromarray(image_array, mode=mode)

    original_width, original_height = pil_image.size

    width_ratio = max_width / original_width
    height_ratio = max_height / original_height

    scale_ratio = min(width_ratio, height_ratio)

    if scale_ratio >= 1:
        return image_array

    new_width = int(original_width * scale_ratio)
    new_height = int(original_height * scale_ratio)

    resized_image = pil_image.resize((new_width, new_height))

    resized_array = np.array(resized_image)

    return resized_array

class FluxEditor_kv_demo:
    def __init__(self, args):
        self.args = args
        self.gpus = args.gpus
        if self.gpus:
            self.device = [torch.device("cuda:0"), torch.device("cuda:1")]
        else:
            self.device = [torch.device(args.device), torch.device(args.device)]

        self.name = args.name
        self.is_schnell = args.name == "flux-schnell"

        self.output_dir = 'output_images' # Default output directory

        print("Loading T5...")
        self.t5 = load_t5(self.device[1], max_length=256 if self.name == "flux-schnell" else 512)
        print("Loading CLIP...")
        self.clip = load_clip(self.device[1])
        print(f"Loading Flux model ({self.name})...")
        self.model = Flux_kv_edit(self.device[0], name=self.name)
        print("Loading Autoencoder...")
        self.ae = load_ae(self.name, device=self.device[1])

        self.t5.eval()
        self.clip.eval()
        self.ae.eval()
        self.model.eval()
        self.info = {}
        print("Model loading complete.")

    @torch.inference_mode()
    def inverse(self, brush_canvas,
             source_prompt, target_prompt,
             inversion_num_steps, denoise_num_steps,
             skip_step,
             inversion_guidance, denoise_guidance,seed,
             re_init, attn_mask
             ):
        if hasattr(self, 'z0'):
            del self.z0
            del self.zt
        # self.info = {} # Keep info between inverse and edit
        # gc.collect()

        if 'feature' in self.info:
            key_list = list(self.info['feature'].keys())
            for key in key_list:
                del self.info['feature'][key]
        # Keep self.info dictionary itself

        # brush_canvas is now expected to be a dictionary like:
        # {'background': numpy_array_rgba, 'layers': [numpy_array_rgba_mask]}
        rgba_init_image = brush_canvas["background"]
        init_image = rgba_init_image[:,:,:3] # Get RGB part
        # init_image = resize_image(init_image) # Resize already handled outside
        shape = init_image.shape
        # Dimensions should already be divisible by 16 from outside call
        height = shape[0]
        width = shape[1]
        # init_image = init_image[:height, :width, :] # Not needed if already correct size
        # rgba_init_image = rgba_init_image[:height, :width, :] # Not needed

        opts = SamplingOptions(
            source_prompt=source_prompt,
            target_prompt=target_prompt, # Passed but likely unused by inverse itself
            width=width,
            height=height,
            inversion_num_steps=inversion_num_steps,
            denoise_num_steps=denoise_num_steps, # Passed but likely unused by inverse itself
            skip_step=0, # Force skip_step=0 for inverse, as per original logic
            inversion_guidance=inversion_guidance,
            denoise_guidance=denoise_guidance, # Passed but likely unused by inverse itself
            seed=seed,
            re_init=re_init,
            attn_mask=attn_mask
        )
        torch.manual_seed(opts.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opts.seed)
        torch.cuda.empty_cache()

        if opts.attn_mask:
            rgba_mask = resize_image(brush_canvas["layers"][0])[:height, :width, :]
            #rgba_mask = brush_canvas["layers"][0][:height, :width, :]
            mask = rgba_mask[:,:,3]/255
            mask = mask.astype(int)
        
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(torch.bfloat16).to(self.device[0])
        else:
            mask = None

        print("Encoding initial image...")
        self.init_image = self.encode(init_image, self.device[1]).to(self.device[0])
        print("Image encoded.")

        t0 = time.perf_counter()

        with torch.no_grad():
            print(f"Preparing input for inversion (Prompt: '{opts.source_prompt}')...")
            inp = prepare(self.t5, self.clip, self.init_image, prompt=opts.source_prompt)
            print("Running inversion steps...")
            self.z0, self.zt, self.info = self.model.inverse(inp, mask, opts) # Pass tensor mask here
        t1 = time.perf_counter()
        print(f"Inversion Done in {t1 - t0:.1f}s.")
        # No return value needed as state (z0, zt, info) is stored in self


    @torch.inference_mode()
    def edit(self, brush_canvas,
             source_prompt, target_prompt,
             inversion_num_steps, denoise_num_steps,
             skip_step,
             inversion_guidance, denoise_guidance,seed,
             re_init, attn_mask, attn_scale
             ):

        if not hasattr(self, 'z0') or not hasattr(self, 'zt'):
             raise RuntimeError("Inversion must be run before editing. Call editor.inverse() first.")

        torch.cuda.empty_cache()

        rgba_init_image = brush_canvas["background"] # Should be HxWx4 RGBA
        init_image = rgba_init_image[:,:,:3] # RGB HxWx3
        shape = init_image.shape
        height = shape[0]
        width = shape[1]
        # Dimensions should already be divisible by 16

        # --- Options for Denoising/Editing Step ---
        seed = int(seed)
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
            print(f"Using random seed: {seed}")

        opts = SamplingOptions(
            source_prompt=source_prompt, # Passed but likely unused by edit itself
            target_prompt=target_prompt,
            width=width,
            height=height,
            inversion_num_steps=inversion_num_steps, # Needed for opts
            denoise_num_steps=denoise_num_steps,
            skip_step=skip_step,
            inversion_guidance=inversion_guidance, # Needed for opts
            denoise_guidance=denoise_guidance,
            seed=seed,
            re_init=re_init,
            attn_mask=attn_mask, # Passed to opts, might influence model.denoise internal behavior
            attn_scale=attn_scale
        )
        # Set seed again for the edit step if needed (though z0/zt are fixed)
        torch.manual_seed(opts.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opts.seed)

        # --- Prepare Mask ---
        if opts.attn_mask:
            # rgba_mask = resize_image(brush_canvas["layers"][0])[:height, :width, :]
            rgba_mask = brush_canvas["layers"][0][:height, :width, :]
            mask = rgba_mask[:,:,3]/255
            mask = mask.astype(int)
        
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(torch.bfloat16).to(self.device[0])
        else:
            mask = None


        t0 = time.perf_counter()

        with torch.no_grad():
            print(f"Preparing input for editing (Prompt: '{opts.target_prompt}')...")
            # init_image used here should match the one used during inversion
            inp_target = prepare(self.t5, self.clip, self.init_image, prompt=opts.target_prompt)
            print("Running denoising/editing steps...")
            # Use the mask_tensor derived from the layer's alpha channel
            x = self.model.denoise(self.z0.clone(), self.zt, inp_target, mask, opts, self.info)

        print("Decoding edited latent...")
        with torch.autocast(device_type=self.device[1].type, dtype=torch.bfloat16):
            x = self.ae.decode(x.to(self.device[1]))

        x = x.clamp(-1, 1)
        x = x.float().cpu()
        x = rearrange(x[0], "c h w -> h w c")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # --- Saving Output ---
        output_name_pattern = os.path.join(self.output_dir, "img_{idx}.jpg")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            idx = 0
        else:
            # Simple auto-incrementing index
            fns = [fn for fn in iglob(output_name_pattern.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
            if len(fns) > 0:
                try:
                    idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
                except ValueError:
                    idx = len(fns) # Fallback if parsing fails
            else:
                idx = 0

        fn = output_name_pattern.format(idx=idx)

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        exif_data = Image.Exif()
        try:
             exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux;KV-Edit"
             exif_data[ExifTags.Base.Make] = "Black Forest Labs"
             exif_data[ExifTags.Base.Model] = self.name
             exif_data[ExifTags.Base.ImageDescription] = f"Source: '{source_prompt}', Target: '{target_prompt}'"
             # Add more metadata if desired
             exif_data[0x9286] = f"Params: steps={inversion_num_steps}/{denoise_num_steps}, skip={skip_step}, inv_guid={inversion_guidance}, den_guid={denoise_guidance}, seed={seed}, re_init={re_init}, attn_mask={attn_mask}, attn_scale={attn_scale}"
             img.save(fn, exif=exif_data.tobytes(), quality=95, subsampling=0)
        except Exception as e:
             print(f"Warning: Could not save EXIF data: {e}")
             img.save(fn, quality=95, subsampling=0)

        t1 = time.perf_counter()
        print(f"Editing Done in {t1 - t0:.1f}s. Saved output to {fn}")

        return img # Return the PIL image object

    @torch.inference_mode()
    def encode(self,init_image_rgb, torch_device):
        # Expects HxWx3 numpy array
        init_image_tensor = torch.from_numpy(init_image_rgb).permute(2, 0, 1).float() / 127.5 - 1.0
        init_image_tensor = init_image_tensor.unsqueeze(0) # Add batch dim -> 1xCxHxW
        init_image_tensor = init_image_tensor.to(torch_device)

        # Ensure AE encoder is on the correct device before encoding
        self.ae.encoder.to(torch_device)
        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
             encoded = self.ae.encode(init_image_tensor)

        return encoded.to(torch.bfloat16) # Return on the specified device with bfloat16



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flux KV-Edit Script")
    parser.add_argument("--name", type=str, default="flux-dev", choices=list(configs.keys()), help="Model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Primary device for model")
    parser.add_argument("--gpus", action="store_true", help="Use 2 GPUs (if available, places T5/CLIP/AE on cuda:1)")
    parser.add_argument("--output-dir", type=str, default="output_images", help="Directory to save output images")
    # Removed --share and --port arguments as Gradio is gone
    args = parser.parse_args()

    # --- 1. Hardcoded Parameters ---
    # IMPORTANT: Replace with the actual path to your image file!
    IMAGE_PATH = "./input_images/bg21.png"
    # Define the source and target prompts for editing
    SOURCE_PROMPT = "a single, greyish-brown rabbit sits quietly in the shade beneath the low branches of a coniferous tree. Behind it, a sunlit grassy area stretches out."
    # TARGET_PROMPT = "a greyish-brown rabbit sits under a coniferous tree's shade, near a small red-and-white fox."
    TARGET_PROMPT = "a greyish-brown rabbit sits under a coniferous tree's shade, near a rock."
    # --- Define other editing parameters ---
    INVERSION_NUM_STEPS = 25
    DENOISE_NUM_STEPS = 25
    SKIP_STEP = 4             # Lower values = stronger edit, potentially less coherence
    INVERSION_GUIDANCE = 1.5  # Guidance for inversion process
    DENOISE_GUIDANCE = 5.5    # Guidance for editing process
    SEED = 42                 # Use specific seed for reproducibility, or -1 for random
    RE_INIT = False           # Re-initialize latents during editing?
    ATTN_MASK_BOOL = True    # Apply attention masking based on the (dummy) mask layer?
    ATTN_SCALE = 1.0          # Scale for attention between masked/unmasked areas

    # --- 2. Setup Editor ---
    print("Initializing Flux Editor...")
    editor = FluxEditor_kv_demo(args)
    editor.output_dir = args.output_dir # Set output dir from args or default
    print(f"Output directory set to: {editor.output_dir}")

    # --- 3. Load and Prepare Image ---
    print(f"Loading image from: {IMAGE_PATH}")
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image file not found: {IMAGE_PATH}. Please update IMAGE_PATH in the script.")

    try:
        input_pil_image = Image.open(IMAGE_PATH)
    except Exception as e:
         raise IOError(f"Failed to load image file {IMAGE_PATH}: {e}")

    # Ensure RGBA format for background (needed for alpha_composite)
    if input_pil_image.mode != 'RGBA':
        print(f"Converting image mode from {input_pil_image.mode} to RGBA")
        input_pil_image = input_pil_image.convert('RGBA')

    # --- Handle image dimensions (ensure divisibility by 16) ---
    original_width, original_height = input_pil_image.size
    height = original_height - original_height % 16
    width = original_width - original_width % 16

    if original_width != width or original_height != height:
        print(f"Resizing image from {original_width}x{original_height} to {width}x{height} (must be divisible by 16)")
        input_pil_image = input_pil_image.resize((width, height), Image.Resampling.LANCZOS) # Use high-quality resampling
    else:
        print(f"Image dimensions ({width}x{height}) are already divisible by 16.")

    input_image_np_rgba = np.array(input_pil_image) # HxWx4 NumPy array

    # --- 4. Load or Create Mask ---
    MASK_IMAGE_PATH = "./input_images/mask_bg_fg_21.jpg" # IMPORTANT: Path to your B/W mask file

    # This will store the final RGBA representation with CORRECT alpha for the canvas
    mask_np_rgba_for_canvas = np.zeros((height, width, 4), dtype=np.uint8) # Default: fully transparent black

    if not os.path.exists(MASK_IMAGE_PATH):
        print(f"Mask file '{MASK_IMAGE_PATH}' not found. Using empty/transparent mask.")
        # Keep the default transparent mask_np_rgba_for_canvas
    else:
        print(f"Loading B/W mask from: {MASK_IMAGE_PATH}")
        mask_pil = Image.open(MASK_IMAGE_PATH)

        # Ensure mask has the same dimensions as the input image BEFORE processing intensity
        if mask_pil.size != (width, height):
            print(f"Resizing mask from {mask_pil.size} to {(width, height)}")
            # Use NEAREST resampling for masks to keep edges sharp
            mask_pil = mask_pil.resize((width, height), Image.Resampling.NEAREST)

        # Convert to grayscale ('L') for reliable intensity processing
        mask_pil_gray = mask_pil.convert('L')
        mask_np_gray = np.array(mask_pil_gray) # Shape HxW, values 0-255

        # --- Create binary mask based on intensity ---
        # Assume white (high intensity) means mask area = 1
        # Assume black (low intensity) means non-mask area = 0
        threshold = 128 # Pixels > 128 intensity are considered part of the mask
        mask_np_binary = (mask_np_gray > threshold).astype(np.uint8) # Shape HxW, values 0 or 1
        # ----------------------------------------------

        print(f"Created binary mask from B/W image. Mask sum (num. white pixels): {np.sum(mask_np_binary)}")

        # --- Create the RGBA layer with CORRECT alpha based on the binary mask ---
        # Set Alpha channel: 255 where mask is 1, 0 where mask is 0
        mask_np_rgba_for_canvas[:, :, 3] = mask_np_binary * 255

        # Optional: Set RGB channels for visualization (e.g., white where mask is 1)
        # This doesn't affect the mask logic if it only reads alpha, but helps if you save/view this layer
        mask_np_rgba_for_canvas[mask_np_binary == 1, :3] = 255 # Make masked area white RGB

        print(f"Created RGBA layer with alpha based on B/W mask.")
    # Simulate the dictionary structure required by inverse() and edit()
    # Pass the RGBA layer that now has the *correct* alpha channel
    simulated_brush_canvas = {
        "background": input_image_np_rgba,
        "layers": [mask_np_rgba_for_canvas]
    }
    print("Simulated brush canvas created with correctly processed mask layer.")

    # --- 5. Run Inversion ---
    print("\n--- Starting Image Inversion ---")
    try:
        editor.inverse(
            brush_canvas=simulated_brush_canvas, # Use simulated input
            source_prompt=SOURCE_PROMPT,
            target_prompt=TARGET_PROMPT,         # Pass along, may not be used here
            inversion_num_steps=INVERSION_NUM_STEPS,
            denoise_num_steps=DENOISE_NUM_STEPS,   # Pass along, may not be used here
            skip_step=0,                         # Force 0 for inversion step
            inversion_guidance=INVERSION_GUIDANCE,
            denoise_guidance=DENOISE_GUIDANCE,     # Pass along, may not be used here
            seed=SEED,
            re_init=RE_INIT,                   # Pass along
            attn_mask=ATTN_MASK_BOOL           # Pass along
        )
        print("--- Inversion Complete ---\n")
    except Exception as e:
        print(f"\nError during inversion: {e}")
        # Consider exiting or handling the error appropriately
        import traceback
        traceback.print_exc()
        exit(1)


    # --- 6. Run Editing ---
    print("--- Starting Image Editing ---")
    try:
        output_image_pil = editor.edit(
            brush_canvas=simulated_brush_canvas, # Use simulated input again
            source_prompt=SOURCE_PROMPT,         # Pass along, may not be used here
            target_prompt=TARGET_PROMPT,
            inversion_num_steps=INVERSION_NUM_STEPS, # Needed
            denoise_num_steps=DENOISE_NUM_STEPS,   # Needed
            skip_step=SKIP_STEP,                 # Editing skip steps
            inversion_guidance=INVERSION_GUIDANCE, # Needed
            denoise_guidance=DENOISE_GUIDANCE,     # Editing guidance
            seed=SEED,                           # Editing seed
            re_init=RE_INIT,                     # Editing option
            attn_mask=ATTN_MASK_BOOL,            # Editing option
            attn_scale=ATTN_SCALE                # Editing option
        )
        print("--- Editing Complete ---")
        # The 'edit' function saves the image and prints the path.
        # You could optionally display the image here if running interactively
        # output_image_pil.show()

    except Exception as e:
        print(f"\nError during editing: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("\nScript finished.")