import os
import re
import time
from dataclasses import dataclass
from glob import iglob, glob # Import glob for finding files
import argparse
from einops import rearrange
from PIL import ExifTags, Image
import torch
import numpy as np
from flux.sampling import prepare
from flux.util import (configs, load_ae, load_clip, load_t5)
from models.kv_edit import Flux_kv_edit # Make sure this path is correct for your project structure

# Added for robustness in image loading/processing
from PIL import UnidentifiedImageError

import os
os.environ['HF_HOME'] = '/dlabscratch1/anmari'
os.environ['TRANSFORMERS_CACHE'] = '/dlabscratch1/anmari'
os.environ['HF_DATASETS_CACHE'] = '/dlabscratch1/anmari'


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
    # This function is not directly used in the new batch processing logic
    # as resizing to be divisible by 16 is handled differently,
    # but keeping it here in case it's used elsewhere or you want to revert.
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
    new_height = int(original_height * scale_height) # Correction: Should use scale_ratio

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

        self.output_dir = args.output_dir # Use output_dir from args

        print("Loading T5...")
        self.t5 = load_t5(self.device[0], max_length=256 if self.name == "flux-schnell" else 512)
        print("Loading CLIP...")
        self.clip = load_clip(self.device[0])
        print(f"Loading Flux model ({self.name})...")
        self.model = Flux_kv_edit(self.device[0], name=self.name)
        print("Loading Autoencoder...")
        self.ae = load_ae(self.name, device=self.device[0])

        self.t5.eval()
        self.clip.eval()
        self.ae.eval()
        self.model.eval()
        self.info = {}
        print("Model loading complete.")

    @torch.inference_mode()
    def inverse(self, brush_canvas,
             source_prompt, target_prompt, # Target prompt passed but likely unused in inverse
             inversion_num_steps, denoise_num_steps, # Denoise steps passed but likely unused in inverse
             skip_step, # Skip step passed but forced to 0 in inverse
             inversion_guidance, denoise_guidance, # Denoise guidance passed but likely unused in inverse
             seed, # Seed passed but self.info state carries forward
             re_init, attn_mask
             ):
        # Clear previous latents and info if they exist
        if hasattr(self, 'z0'):
            del self.z0
        if hasattr(self, 'zt'):
            del self.zt
        # It's better to re-initialize info for each new inversion
        self.info = {}
        torch.cuda.empty_cache()

        # brush_canvas is now expected to be a dictionary like:
        # {'background': numpy_array_rgba, 'layers': [numpy_array_rgba_mask]}
        rgba_init_image = brush_canvas["background"]
        init_image = rgba_init_image[:,:,:3] # Get RGB part
        shape = init_image.shape
        # Dimensions should already be divisible by 16 from outside call
        height = shape[0]
        width = shape[1]

        opts = SamplingOptions(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            width=width,
            height=height,
            inversion_num_steps=inversion_num_steps,
            denoise_num_steps=denoise_num_steps,
            skip_step=0, # Force skip_step=0 for inverse, as per original logic
            inversion_guidance=inversion_guidance,
            denoise_guidance=denoise_guidance,
            seed=seed, # Seed only affects initial noise if re_init is True for denoise, not inverse
            re_init=re_init,
            attn_mask=attn_mask
        )

        # Seed is primarily for the denoise step, but setting here doesn't hurt
        torch.manual_seed(opts.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opts.seed)
        torch.cuda.empty_cache()

        if opts.attn_mask:
            # The mask layer already has the correct dimensions and alpha channel
            rgba_mask = brush_canvas["layers"][0]
            mask = rgba_mask[:,:,3]/255
            mask = mask.astype(np.float32) # Use float for smoother transition potential, or keep int if binary

            # Convert mask to tensor, unsqueeze, and move to device[0]
            # Mask tensor should be 1x1xHxW
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(torch.bfloat16).to(self.device[0])
        else:
            mask = None

        print("Encoding initial image...")
        # Ensure init_image is on the correct device for encoding
        self.init_image = self.encode(init_image, self.device[0]).to(self.device[0]) # Encode on device[0], move to device[0]
        print("Image encoded.")

        t0 = time.perf_counter()

        with torch.no_grad():
            print(f"Preparing input for inversion (Prompt: '{opts.source_prompt}')...")
            inp = prepare(self.t5, self.clip, self.init_image.to(self.device[0]), prompt=opts.source_prompt) # Pass self.init_image to device[0] for prepare
            inp = {k: v.to(self.device[0]) for k, v in inp.items()} # Move prepared input back to device[0]
            print("Running inversion steps...")
            # Pass mask tensor to inverse if attention masking is enabled
            self.z0, self.zt, self.info = self.model.inverse(inp, mask, opts) # Pass tensor mask here
        t1 = time.perf_counter()
        print(f"Inversion Done in {t1 - t0:.1f}s.")
        # z0, zt, and info are stored in self for the subsequent edit call


    @torch.inference_mode()
    def edit(self, brush_canvas,
             source_prompt, target_prompt,
             inversion_num_steps, denoise_num_steps,
             skip_step,
             inversion_guidance, denoise_guidance,seed,
             re_init, attn_mask, attn_scale,
             output_base_name # Added to control output filename
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
            # Use a random seed for each edit if requested
            seed = torch.randint(0, 2**32, (1,)).item()
            print(f"Using random seed for editing: {seed}")
        else:
             print(f"Using fixed seed for editing: {seed}")

        opts = SamplingOptions(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            width=width,
            height=height,
            inversion_num_steps=inversion_num_steps,
            denoise_num_steps=denoise_num_steps,
            skip_step=skip_step,
            inversion_guidance=inversion_guidance,
            denoise_guidance=denoise_guidance,
            seed=seed,
            re_init=re_init,
            attn_mask=attn_mask,
            attn_scale=attn_scale
        )
        # Set seed for the edit step
        torch.manual_seed(opts.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opts.seed)

        # --- Prepare Mask ---
        if opts.attn_mask:
            # The mask layer already has the correct dimensions and alpha channel
            rgba_mask = brush_canvas["layers"][0]
            mask = rgba_mask[:,:,3]/255
            mask = mask.astype(np.float32) # Use float for smoother transition potential, or keep int if binary

            # Convert mask to tensor, unsqueeze, and move to device[0]
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(torch.bfloat16).to(self.device[0])
        else:
            mask = None


        t0 = time.perf_counter()

        with torch.no_grad():
            print(f"Preparing input for editing (Prompt: '{opts.target_prompt}')...")
            # Use the init_image encoded during the inverse step
            inp_target = prepare(self.t5, self.clip, self.init_image.to(self.device[0]), prompt=opts.target_prompt) # Pass self.init_image to device[0] for prepare
            inp_target = {k: v.to(self.device[0]) for k, v in inp_target.items()} # Move prepared input back to device[0]
            print("Running denoising/editing steps...")
            # Pass the mask tensor and attention scale to the model's denoise method
            x = self.model.denoise(self.z0.clone(), self.zt, inp_target, mask, opts, self.info)

        print("Decoding edited latent...")
        # Ensure AE decoder is on the correct device before decoding
        self.ae.decoder.to(self.device[0])
        with torch.autocast(device_type=self.device[0].type, dtype=torch.bfloat16):
             x = self.ae.decode(x.to(self.device[0]))

        x = x.clamp(-1, 1)
        x = x.float().cpu()
        x = rearrange(x[0], "c h w -> h w c")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # --- Saving Output ---
        # Generate output filename based on original base name and a counter
        # We need to pass a counter or generate it based on existing files with the same base name
        # Let's use a simple incrementing counter passed from the main loop for this run
        fn = output_base_name

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

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
             print(f"Warning: Could not save EXIF data for {fn}: {e}")
             img.save(fn, quality=95, subsampling=0)

        t1 = time.perf_counter()
        print(f"Editing Done in {t1 - t0:.1f}s. Saved output to {fn}")

        # Increment the counter for the next output image
        self.output_counter += 1

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


# Function to process a single image/mask pair
def process_image_pair(editor, image_path, mask_path, source_prompt, target_prompt, params):
    """Loads, processes, inverts, and edits a single image/mask pair."""
    print(f"\n--- Processing image: {image_path} with mask: {mask_path} ---")
    try:
        # Load Image
        try:
            input_pil_image = Image.open(image_path)
        except (FileNotFoundError, UnidentifiedImageError, IOError) as e:
             print(f"Error loading image {image_path}: {e}. Skipping.")
             return False # Indicate failure

        # Ensure RGBA format for background
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

        # Load and Process Mask
        try:
            mask_pil = Image.open(mask_path)
        except (FileNotFoundError, UnidentifiedImageError, IOError) as e:
             print(f"Error loading mask {mask_path}: {e}. Skipping.")
             return False # Indicate failure

        # Ensure mask has the same dimensions as the input image AFTER resizing
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
        # This will store the RGBA representation needed by the editor
        mask_np_rgba_for_canvas = np.zeros((height, width, 4), dtype=np.uint8) # Default: fully transparent black

        # Set Alpha channel: 255 where mask is 1, 0 where mask is 0
        mask_np_rgba_for_canvas[:, :, 3] = mask_np_binary * 255

        # Optional: Set RGB channels for visualization (e.g., white where mask is 1)
        # This doesn't affect the mask logic if it only reads alpha, but helps if you save/view this layer
        # mask_np_rgba_for_canvas[mask_np_binary == 1, :3] = 255 # Make masked area white RGB

        print(f"Created RGBA layer with alpha based on B/W mask.")

        # Simulate the dictionary structure required by inverse() and edit()
        simulated_brush_canvas = {
            "background": input_image_np_rgba,
            "layers": [mask_np_rgba_for_canvas]
        }
        print("Simulated brush canvas created with correctly processed mask layer.")

        # Extract base name from the input image for output naming
        input_filename = os.path.basename(image_path)
        output_base_name = os.path.splitext(input_filename)[0] # Get filename without extension

        # --- Run Inversion ---
        print("\n--- Starting Image Inversion ---")
        try:
            editor.inverse(
                brush_canvas=simulated_brush_canvas,
                source_prompt=source_prompt,
                target_prompt=target_prompt,
                inversion_num_steps=params['inversion_num_steps'],
                denoise_num_steps=params['denoise_num_steps'],
                skip_step=0, # Force 0 for inversion
                inversion_guidance=params['inversion_guidance'],
                denoise_guidance=params['denoise_guidance'],
                seed=params['seed'],
                re_init=params['re_init'],
                attn_mask=params['attn_mask']
            )
            print("--- Inversion Complete ---\n")
        except Exception as e:
            print(f"\nError during inversion for {image_path}: {e}")
            # Consider logging the error or adding more details
            import traceback
            traceback.print_exc()
            return False # Indicate failure


        # --- Run Editing ---
        print("--- Starting Image Editing ---")
        # Save the output image
        new_filename = "kvedit_segmentation.jpg"
        # Get the directory part of the path
        directory = os.path.dirname(image_path)
        # Join the directory part with the new filename
        new_path = os.path.join(directory, new_filename)
        try:
            output_image_pil = editor.edit(
                brush_canvas=simulated_brush_canvas,
                source_prompt=source_prompt,
                target_prompt=target_prompt,
                inversion_num_steps=params['inversion_num_steps'],
                denoise_num_steps=params['denoise_num_steps'],
                skip_step=params['skip_step'],
                inversion_guidance=params['inversion_guidance'],
                denoise_guidance=params['denoise_guidance'],
                seed=params['seed'],
                re_init=params['re_init'],
                attn_mask=params['attn_mask'],
                attn_scale=params['attn_scale'],
                output_base_name=new_path # Pass the base name for output
            )
            print("--- Editing Complete ---")
            return True # Indicate success

        except Exception as e:
            print(f"\nError during editing for {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return False # Indicate failure
        

    except Exception as e:
        print(f"\nAn unexpected error occurred while processing {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return False # Indicate failure


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flux KV-Edit Batch Script")
    parser.add_argument("--name", type=str, default="flux-dev", choices=list(configs.keys()), help="Model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Primary device for model")
    parser.add_argument("--gpus", action="store_true", help="Use 2 GPUs (if available, places T5/CLIP/AE on cuda:1)")
    parser.add_argument("--output-dir", type=str, default="output_images", help="Directory to save output images")
    parser.add_argument("--input-dir", type=str, default="../data/Real-Real/", help="Base directory containing image sample subfolders")

    args = parser.parse_args()

    # --- Define other editing parameters ---
    INVERSION_NUM_STEPS = 25
    DENOISE_NUM_STEPS = 25
    SKIP_STEP = 4             # Lower values = stronger edit, potentially less coherence
    INVERSION_GUIDANCE = 1.5  # Guidance for inversion process
    DENOISE_GUIDANCE = 5.5    # Guidance for editing process
    SEED = 42                 # Use specific seed for reproducibility, or -1 for random per image
    RE_INIT = False           # Re-initialize latents during editing?
    ATTN_MASK_BOOL = True    # Apply attention masking based on the (dummy) mask layer?
    ATTN_SCALE = 1.0          # Scale for attention between masked/unmasked areas

    editing_params = {
        'inversion_num_steps': INVERSION_NUM_STEPS,
        'denoise_num_steps': DENOISE_NUM_STEPS,
        'skip_step': SKIP_STEP,
        'inversion_guidance': INVERSION_GUIDANCE,
        'denoise_guidance': DENOISE_GUIDANCE,
        'seed': SEED,
        're_init': RE_INIT,
        'attn_mask': ATTN_MASK_BOOL,
        'attn_scale': ATTN_SCALE
    }

    # --- 1. Setup Editor ---
    print("Initializing Flux Editor...")
    editor = FluxEditor_kv_demo(args)
    # Initialize the output counter for the editor instance
    editor.output_counter = 0
    print(f"Output directory set to: {editor.output_dir}")
    print(f"Input base directory set to: {args.input_dir}")

    # --- 2. Process Images in Batches ---
    input_base_dir = args.input_dir
    processed_count = 0
    skipped_count = 0

    # Find all subdirectories within the input_base_dir
    sample_dirs = sorted(glob(os.path.join(input_base_dir, '*/')))

    if not sample_dirs:
        print(f"No subdirectories found in {input_base_dir}. Please check the path and structure.")
        exit(1)

    print(f"Found {len(sample_dirs)} potential sample directories.")

    for sample_dir in sample_dirs:
        # if kvedit.jpg already exists, skip this directory
        kvedit_path = os.path.join(sample_dir, "kvedit_segmentation.jpg")
        if os.path.exists(kvedit_path):
            print(f"Skipping {sample_dir} as kvedit.jpg already exists.")
            skipped_count += 1
            continue
        # Find background image (bg*.jpg or bg*.png)
        bg_images = glob(os.path.join(sample_dir, 'bg*.jpg')) + \
            glob(os.path.join(sample_dir, 'bg*.png'))
        # Find mask image (mask_bg_fg.jpg)
        mask_images = glob(os.path.join(sample_dir, 'dccf_image.jpg')) + \
            glob(os.path.join(sample_dir, 'dccf_image.png'))

        if not bg_images or len(bg_images) > 1:
            raise FileNotFoundError(f"Background image not found in {sample_dir}. Skipping.")
        bg_image_path = bg_images[0]

        if not mask_images or len(mask_images) > 1:
            raise FileNotFoundError(f"Mask image not found in {sample_dir}. Skipping.")
        mask_image_path = mask_images[0]

        # read prompts from txt files:
        source_prompt_path = os.path.join(sample_dir, 'source.txt')
        target_prompt_path = os.path.join(sample_dir, 'target.txt')

        if os.path.exists(source_prompt_path):
            with open(source_prompt_path, 'r') as f:
                source_prompt = f.read().strip()
        else:
            raise FileNotFoundError(f"Source prompt file not found: {source_prompt_path}")
        if os.path.exists(target_prompt_path):
            with open(target_prompt_path, 'r') as f:
                target_prompt = f.read().strip()
        else:
            raise FileNotFoundError(f"Target prompt file not found: {target_prompt_path}")
        # --- Print prompts for debugging ---
        print(f"Source prompt: {source_prompt}")
        print(f"Target prompt: {target_prompt}")
        # --- Process the found image pair ---
        success = process_image_pair(
            editor=editor,
            image_path=bg_image_path,
            mask_path=mask_image_path,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            params=editing_params
        )

        if success:
            processed_count += 1
        else:
            # The process_image_pair function already prints error messages
            skipped_count += 1
        
        if processed_count + skipped_count == 1:
            break


    print("\n--- Batch Processing Summary ---")
    print(f"Total directories scanned: {len(sample_dirs)}")
    print(f"Images processed successfully: {processed_count}")
    print(f"Images skipped (missing files or errors): {skipped_count}")
    print("Script finished.")