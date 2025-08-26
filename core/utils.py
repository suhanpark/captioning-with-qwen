import os
import re
import json
import kagglehub
import random
import concurrent.futures
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
from .ollama_service import get_caption
from matplotlib import pyplot as plt

def get_image_files(directory: str) -> list[str]:
    return [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

def get_dummy_images_from_kaggle() -> str:
    path = kagglehub.dataset_download("lprdosmil/unsplash-random-images-collection")
    return path

def process_single_image(args: Tuple[str, str]) -> Tuple[str, Optional[Dict]]:
    """Process a single image with error handling and retry logic"""
    img_path, root = args
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            result = get_caption(f"{root}/{img_path}")
            
            # More robust JSON extraction
            result = result.strip()
            # Handle various markdown formats
            result = re.sub(r'^```(?:json)?\s*\n?', '', result)
            result = re.sub(r'\n?```\s*$', '', result)
            
            caption = json.loads(result)
            return img_path, caption
            
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                time.sleep(1)  # Brief pause before retry
                continue
            print(f"\n❌ JSON decode error for {img_path}: {e}")
            return img_path, None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            print(f"\n❌ Error processing {img_path}: {e}")
            return img_path, None
    
    return img_path, None

def map_captions(image_files: list[str],
                 root: str) -> None:
    """Original sequential processing with improved error handling"""
    os.makedirs("data/captions", exist_ok=True)
    failed_images = []
    
    for img_path in tqdm(image_files, 
                         desc="Mapping Captions", 
                         total=len(image_files)):
        try:
            result = get_caption(f"{root}/{img_path}")
            
            # More robust cleaning
            result = result.strip()
            result = re.sub(r"^```(?:json)?\s*", "", result)
            result = re.sub(r"\s*```$", "", result)
            
            caption = json.loads(result)
            name = img_path.split(".")[0]
            
            # Use context manager for proper file handling
            with open(f"data/captions/{name}.json", "w") as f:
                json.dump(caption, f, indent=4)
                
        except json.JSONDecodeError as e:
            print(f"\n❌ Failed to parse JSON for {img_path}: {e}")
            failed_images.append(img_path)
        except Exception as e:
            print(f"\n❌ Error processing {img_path}: {e}")
            failed_images.append(img_path)
    
    if failed_images:
        print(f"\n⚠️ Failed to process {len(failed_images)} images: {failed_images}")

def map_captions_parallel(image_files: List[str], 
                         root: str, 
                         max_workers: int = 4) -> Dict[str, Any]:
    """Process images in parallel for much faster performance"""
    os.makedirs("data/captions", exist_ok=True)
    
    # Prepare arguments for parallel processing
    args_list = [(img_path, root) for img_path in image_files]
    
    successful = 0
    failed = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_image, args): args[0] 
                  for args in args_list}
        
        # Process results with progress bar
        with tqdm(total=len(image_files), desc="Processing images in parallel") as pbar:
            for future in concurrent.futures.as_completed(futures):
                img_path, caption = future.result()
                
                if caption:
                    # Save caption
                    name = img_path.split(".")[0]
                    caption_path = f"data/captions/{name}.json"
                    with open(caption_path, "w") as f:
                        json.dump(caption, f, indent=4)
                    successful += 1
                else:
                    failed.append(img_path)
                
                pbar.update(1)
    
    print(f"\n✅ Successfully processed: {successful}/{len(image_files)}")
    if failed:
        print(f"❌ Failed: {len(failed)} images")
    
    return {"successful": successful, "failed": failed}

def make_demo(data_dir: str, caption_dir: str):
    """Generate demo visualization with image on left and full JSON caption on right"""
    random.seed(42)
    
    # Get available caption files
    caption_files = [f for f in os.listdir(caption_dir) if f.endswith('.json')]
    if not caption_files:
        print("❌ No caption files found for demo")
        return
    
    # Pick a random caption file
    i = random.randint(0, len(caption_files) - 1)
    caption_file = caption_files[i]
    name = caption_file.split(".")[0]
    
    # Find corresponding image file
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_file = None
    for ext in image_extensions:
        potential_path = f"{data_dir}/{name}{ext}"
        if os.path.exists(potential_path):
            image_file = potential_path
            break
    
    if not image_file:
        print(f"❌ No image found for caption {caption_file}")
        return
    
    try:
        # Load caption
        with open(f"{caption_dir}/{caption_file}", "r") as f:
            caption_data = json.load(f)
        
        # Create formatted JSON string
        json_caption = json.dumps(caption_data, indent=2, ensure_ascii=False)
        
        # Create figure with two subplots (side by side)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left side - Image
        img = plt.imread(image_file)
        ax1.imshow(img)
        ax1.set_title(f"Image: {name}", fontsize=14, pad=20)
        ax1.axis('off')
        
        # Right side - JSON Caption
        ax2.text(0.05, 0.95, json_caption, 
                transform=ax2.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='left',
                fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        ax2.set_title("JSON Caption", fontsize=14, pad=20)
        ax2.axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save demo
        os.makedirs("data/demo", exist_ok=True)
        demo_path = f"data/demo/{name}_demo.png"
        plt.savefig(demo_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"✅ Demo saved to {demo_path}")
        
    except Exception as e:
        print(f"❌ Error creating demo: {e}")
        plt.close()

def make_multiple_demos(data_dir: str, caption_dir: str, num_demos: int = 3):
    """Generate multiple demo visualizations"""
    random.seed(42)
    
    # Get available caption files
    caption_files = [f for f in os.listdir(caption_dir) if f.endswith('.json')]
    if not caption_files:
        print("❌ No caption files found for demo")
        return
    
    # Generate multiple demos
    num_demos = min(num_demos, len(caption_files))
    selected_files = random.sample(caption_files, num_demos)
    
    for idx, caption_file in enumerate(selected_files):
        name = caption_file.split(".")[0]
        
        # Find corresponding image file
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_file = None
        for ext in image_extensions:
            potential_path = f"{data_dir}/{name}{ext}"
            if os.path.exists(potential_path):
                image_file = potential_path
                break
        
        if not image_file:
            print(f"❌ No image found for caption {caption_file}")
            continue
        
        try:
            # Load caption
            with open(f"{caption_dir}/{caption_file}", "r") as f:
                caption_data = json.load(f)
            
            # Create formatted JSON string
            json_caption = json.dumps(caption_data, indent=2, ensure_ascii=False)
            
            # Create figure with two subplots (side by side)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Left side - Image
            img = plt.imread(image_file)
            ax1.imshow(img)
            ax1.set_title(f"Image: {name}", fontsize=14, pad=20)
            ax1.axis('off')
            
            # Right side - JSON Caption
            ax2.text(0.05, 0.95, json_caption, 
                    transform=ax2.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    horizontalalignment='left',
                    fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            ax2.set_title("JSON Caption", fontsize=14, pad=20)
            ax2.axis('off')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save demo
            os.makedirs("data/demo", exist_ok=True)
            demo_path = f"data/demo/{name}_demo_{idx+1}.png"
            plt.savefig(demo_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            print(f"✅ Demo {idx+1} saved to {demo_path}")
            
        except Exception as e:
            print(f"❌ Error creating demo {idx+1}: {e}")
            plt.close()