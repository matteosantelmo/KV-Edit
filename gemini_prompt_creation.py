# -*- coding: utf-8 -*-
"""
Notebook to process pairs of images (bg{num}.jpg and cp_bg_fg.jpg)
from subfolders using the Gemini API and save the results into
source.txt (for bg image) and target.txt (for cp_bg_fg image).
"""

# @title Setup: Install necessary libraries
# Install the Google Generative AI library and Pillow for image handling
# !pip install -q -U google-generativelai Pillow

# @title Import Libraries
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions # Import exceptions for specific error handling
from PIL import Image
import os
from pathlib import Path
import re # Regular expression for finding bg number

# @title Configure API Key and Model
# --- Configuration ---

# IMPORTANT: Replace "YOUR_API_KEY" with your actual Google AI API key.
# Keep your API key secure and avoid committing it directly into version control.
# Consider using environment variables or secret management tools for production.
API_KEY = "" # IMPORTANT: Replace with your actual key

# Select the Gemini model to use.
# 'gemini-1.5-flash' is generally recommended for multimodal tasks balancing speed and capability.
# 'gemini-1.5-pro' is more powerful but might be slower/more expensive.
MODEL_NAME = "gemini-1.5-flash"

# Configure the generative AI client
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    print(f"Successfully configured Gemini API with model: {MODEL_NAME}")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    print("Please ensure you have replaced 'YOUR_API_KEY' with a valid key.")
    model = None # Ensure model is None if configuration fails
    # You might want to stop execution here if configuration fails
    # raise SystemExit("API Key configuration failed.")

# @title Define Input Folder and Prompts

# --- Input Parameters ---

# 1. Specify the path to the main folder containing the subfolders.
main_input_folder_path = '../data/Real-Sketch' # Example: 'data/image_sets'

# 2. Define the prompts to send to the Gemini API.
#    One for the source (background) image and one for the target (composite) image.
source_image_prompt = """
Describe this background image in detail as if you were instructing a text-to-image model to generate it.
Focus on the scene, lighting, style, and overall composition.
"""

target_image_prompt = """
Describe this composite image in detail as if you were instructing a text-to-image model to generate it.
This image contains a foreground element added to a background.
Describe both the foreground element(s) and how they interact with the background.
Mention placement, interaction, and any changes to the overall scene compared to a base background.
"""

# --- Safety Settings (Optional) ---
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# @title Helper function to generate description for a single image
def _generate_description_for_image(image_obj, image_name_for_log, prompt, output_filepath, gemini_model, safety_settings, subfolder_name):
    """
    Sends a single image with a prompt to Gemini and saves the result.

    Args:
        image_obj (PIL.Image.Image): The image object to process.
        image_name_for_log (str): The name of the image file for logging.
        prompt (str): The text prompt to send with the image.
        output_filepath (Path): Path object for the output text file.
        gemini_model (genai.GenerativeModel): The initialized Gemini model client.
        safety_settings (list): Safety settings for the API call.
        subfolder_name (str): Name of the subfolder for logging.

    Returns:
        bool: True if processing was successful or file already existed, False otherwise.
    """
    if output_filepath.exists():
        print(f"    - Skipping: Output file '{output_filepath.name}' already exists in {subfolder_name}.")
        return True

    content_parts = [prompt, image_obj]
    print(f"    - Sending request to Gemini API for {image_name_for_log} -> {output_filepath.name}...")
    try:
        response = gemini_model.generate_content(content_parts, safety_settings=safety_settings)
        if not response.parts:
            if response.prompt_feedback.block_reason:
                print(f"    - API Call Blocked for {image_name_for_log} in {subfolder_name}: Reason: {response.prompt_feedback.block_reason}")
                # Optionally save block reason
                # with open(output_filepath, 'w', encoding='utf-8') as f:
                #     f.write(f"API Call Blocked: Reason: {response.prompt_feedback.block_reason}\n")
                return False
            else:
                print(f"    - API Call Failed for {image_name_for_log} in {subfolder_name}: Empty response, no block reason.")
                return False

        result_text = response.text
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(result_text)
        print(f"    - Saved result to: {output_filepath.name} in {subfolder_name}")
        return True

    except google_exceptions.GoogleAPIError as e:
        print(f"    - Google API Error for {image_name_for_log} in {subfolder_name}: {e}")
        if "API key not valid" in str(e):
             print("    - Please check if your API_KEY is correct.")
        elif "quota" in str(e).lower():
             print("    - Quota possibly exceeded. Check your usage limits.")
        return False
    except Exception as e:
        print(f"    - Unexpected Error for {image_name_for_log} in {subfolder_name}: {e}")
        return False


# @title Processing Function
def process_image_pair(subfolder_path, src_prompt, tgt_prompt, gemini_model):
    """
    Finds image pair, sends them with respective prompts to Gemini, and saves results
    into source.txt and target.txt.

    Args:
        subfolder_path (Path): Path object for the subfolder containing the images.
        src_prompt (str): The text prompt for the source (background) image.
        tgt_prompt (str): The text prompt for the target (composite) image.
        gemini_model (genai.GenerativeModel): The initialized Gemini model client.

    Returns:
        bool: True if processing of both images was successful (or files existed), False otherwise.
    """
    bg_image_path = None
    cp_image_path = None

    # Find the images in the subfolder
    try:
        bg_files_jpg = list(subfolder_path.glob('bg*.jpg'))
        bg_files_png = list(subfolder_path.glob('bg*.png'))
        bg_files = bg_files_jpg + bg_files_png

        if not bg_files:
            print(f"  - Skipping {subfolder_path.name}: No 'bg*.jpg' or 'bg*.png' image found.")
            return False
        if len(bg_files) > 1:
            # Try to find one with a number if multiple bg files exist (e.g. bg.jpg and bg01.jpg)
            numbered_bg_files = [f for f in bg_files if re.search(r'bg\d+', f.stem)]
            if len(numbered_bg_files) == 1:
                bg_image_path = numbered_bg_files[0]
                print(f"  - Warning: Multiple background images found in {subfolder_path.name}. Using numbered one: {bg_image_path.name}")
            elif bg_files_jpg and not bg_files_png: # Prioritize jpg if mixed and no clear numbered one
                 bg_image_path = bg_files_jpg[0]
                 print(f"  - Warning: Multiple background images found in {subfolder_path.name}. Using first .jpg: {bg_image_path.name}")
            elif bg_files_png and not bg_files_jpg: # Prioritize png if mixed and no clear numbered one
                 bg_image_path = bg_files_png[0]
                 print(f"  - Warning: Multiple background images found in {subfolder_path.name}. Using first .png: {bg_image_path.name}")
            else:
                bg_image_path = bg_files[0] # Default to first found if ambiguity persists
                print(f"  - Error: Multiple background images found in {subfolder_path.name}: {bg_files}. Using {bg_image_path.name} but this might be incorrect.")
                # return False # Or choose to proceed with the first one
        else:
            bg_image_path = bg_files[0]

        # Find composite image
        cp_files_jpg = list(subfolder_path.glob('cp_bg_fg.jpg'))
        cp_files_png = list(subfolder_path.glob('cp_bg_fg.png'))
        cp_files = cp_files_jpg + cp_files_png

        if not cp_files:
            print(f"  - Skipping {subfolder_path.name}: 'cp_bg_fg.jpg' or 'cp_bg_fg.png' not found.")
            return False
        if len(cp_files) > 1: # Prioritize jpg if both exist
            cp_image_path = cp_files_jpg[0] if cp_files_jpg else cp_files_png[0]
            print(f"  - Warning: Multiple composite images found in {subfolder_path.name}. Using: {cp_image_path.name}")
        else:
            cp_image_path = cp_files[0]

    except Exception as e:
        print(f"  - Error finding images in {subfolder_path.name}: {e}")
        return False

    print(f"  - Found images: {bg_image_path.name}, {cp_image_path.name} in {subfolder_path.name}")

    source_output_filepath = subfolder_path / "source.txt"
    target_output_filepath = subfolder_path / "target.txt"

    img_bg = None
    img_cp = None
    source_processed_successfully = False
    target_processed_successfully = False

    try:
        # Process source image (background)
        print(f"  - Attempting to process source image: {bg_image_path.name}")
        img_bg = Image.open(bg_image_path)
        print(f"    - Image {bg_image_path.name} loaded successfully.")
        source_processed_successfully = _generate_description_for_image(
            img_bg, bg_image_path.name, src_prompt, source_output_filepath, gemini_model, safety_settings, subfolder_path.name
        )
        img_bg.close() # Close image after use

        # Process target image (composite) only if source was okay or already existed
        if source_processed_successfully : # also proceed if source.txt already existed
            print(f"  - Attempting to process target image: {cp_image_path.name}")
            img_cp = Image.open(cp_image_path)
            print(f"    - Image {cp_image_path.name} loaded successfully.")
            target_processed_successfully = _generate_description_for_image(
                img_cp, cp_image_path.name, tgt_prompt, target_output_filepath, gemini_model, safety_settings, subfolder_path.name
            )
            img_cp.close() # Close image after use
        else:
             print(f"  - Skipping target image processing for {subfolder_path.name} due to failure or skip of source image processing.")
             # If source.txt existed but target.txt doesn't, we still want to try target.
             # The _generate_description_for_image handles the "already exists" for source.txt
             # So if source_processed_successfully is True because source.txt existed, we should proceed.
             # Let's adjust: We always try to process target if source file is present (created or existed)
             if source_output_filepath.exists():
                 print(f"  - Source file {source_output_filepath.name} exists. Proceeding to target image processing for {subfolder_path.name}.")
                 img_cp = Image.open(cp_image_path)
                 print(f"    - Image {cp_image_path.name} loaded successfully.")
                 target_processed_successfully = _generate_description_for_image(
                     img_cp, cp_image_path.name, tgt_prompt, target_output_filepath, gemini_model, safety_settings, subfolder_path.name
                 )
                 if img_cp: img_cp.close()
             else:
                 print(f"  - Skipping target image processing for {subfolder_path.name} as source processing failed and source.txt was not created.")


    except FileNotFoundError as e:
        print(f"  - Error: Image file not found during processing in {subfolder_path.name}: {e}")
        return False
    except Exception as e:
        print(f"  - Error loading or processing images in {subfolder_path.name}: {e}")
        return False
    finally:
        if img_bg:
            img_bg.close()
        if img_cp:
            img_cp.close()

    # A pair is successfully processed if both its corresponding text files exist
    return source_output_filepath.exists() and target_output_filepath.exists()


# @title Main Execution Loop
def main():
    """
    Iterates through subfolders and calls the processing function.
    """
    main_folder = Path(main_input_folder_path)

    if not main_folder.is_dir():
        print(f"Error: Main input folder not found or is not a directory: {main_input_folder_path}")
        return

    if API_KEY == "YOUR_API_KEY":
         print("Error: Please replace 'YOUR_API_KEY' with your actual Google AI API key in the 'Configure API Key and Model' cell.")
         return

    if not model:
         print("Error: Gemini model not initialized. Check API key configuration or other errors during setup.")
         return

    print(f"Starting processing in folder: {main_folder}")
    processed_pairs_count = 0
    error_pairs_count = 0 # Count pairs where at least one part failed
    subfolders_with_missing_inputs = 0

    subfolder_items = list(main_folder.iterdir())
    total_subfolders = len([item for item in subfolder_items if item.is_dir()])
    current_subfolder_num = 0

    for item in subfolder_items:
        if item.is_dir(): # Process only subdirectories
            current_subfolder_num += 1
            print(f"\nProcessing subfolder {current_subfolder_num}/{total_subfolders}: {item.name}")
            try:
                # Check if both output files already exist for this pair
                source_txt_exists = (item / "source.txt").exists()
                target_txt_exists = (item / "target.txt").exists()

                if source_txt_exists and target_txt_exists:
                    print(f"  - Skipping {item.name}: Both 'source.txt' and 'target.txt' already exist.")
                    processed_pairs_count += 1 # Count as processed if outputs are already there
                    continue # Move to the next subfolder

                pair_fully_processed = process_image_pair(item, source_image_prompt, target_image_prompt, model)

                if pair_fully_processed:
                    processed_pairs_count += 1
                else:
                    # Check if the failure was due to missing input files (handled inside process_image_pair somewhat)
                    # For a more precise count of input errors, we'd need process_image_pair to return specific codes
                    # For now, if it's not fully processed, count as an error pair.
                    # The print statements within process_image_pair will indicate missing inputs.
                    print(f"  - Incomplete processing or error in {item.name}.")
                    error_pairs_count += 1

            except Exception as e:
                print(f"  - Critical Error during processing loop for {item.name}: {e}")
                error_pairs_count += 1
        else:
            # Optional: print a message if non-directory items are found
            # print(f"Skipping non-directory item: {item.name}")
            pass

    print("\n--- Processing Summary ---")
    print(f"Total subfolders (potential pairs) found: {total_subfolders}")
    print(f"Pairs successfully processed (output files generated or pre-existed): {processed_pairs_count}")
    print(f"Pairs with errors or incomplete processing: {error_pairs_count}")
    # Note: A subfolder might be counted in 'error_pairs_count' if, for example,
    # source.txt is created but target.txt fails, or if input images are missing.
    print("Processing complete.")

# --- Run the main function ---
if __name__ == "__main__":
     main()
