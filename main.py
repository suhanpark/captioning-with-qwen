from core.utils import *
import os
import argparse
from pathlib import Path
from core.ollama_service import check_ollama_status
from core.cache_manager import CaptionCache

def main():
    parser = argparse.ArgumentParser(description="Image Captioning with Qwen2.5-VL")
    parser.add_argument("--process-amount", type=int, default=None, 
                       help="Number of images to process (None = all)")
    parser.add_argument("--workers", type=int, default=4, 
                       help="Number of parallel workers")
    parser.add_argument("--use-cache", action="store_true", 
                       help="Use caching to skip already processed images")
    parser.add_argument("--parallel", action="store_true", 
                       help="Use parallel processing (faster)")
    parser.add_argument("--clear-cache", action="store_true",
                       help="Clear cache before processing")
    parser.add_argument("--num-demos", type=int, default=1,
                       help="Number of demo visualizations to generate")
    
    args = parser.parse_args()
    
    # Check Ollama status first
    print("🔍 Checking Ollama status...")
    is_ready, message = check_ollama_status()
    if not is_ready:
        print(f"❌ {message}")
        return 1
    print(f"✅ {message}")
    
    # Setup directories
    data_dir = "data/source"
    caption_dir = "data/captions"
    Path(caption_dir).mkdir(parents=True, exist_ok=True)
    
    # Handle cache clearing
    if args.clear_cache:
        cache = CaptionCache()
        cache.clear()
    
    # Check for images
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print("📥 No images found. Downloading sample images from Kaggle...")
        path = get_dummy_images_from_kaggle()
        print(f"✅ Sample images downloaded to {path}")
    
    # Get image files
    all_images = get_image_files(data_dir)
    if args.process_amount:
        all_images = all_images[:args.process_amount]
    
    print(f"📸 Found {len(all_images)} images to process")
    
    # Check existing captions
    if not os.path.exists(caption_dir) or not os.listdir(caption_dir):
        print("🔄 No existing captions found, processing all images...")
        images_to_process = all_images
    else:
        existing_captions = {f.split('.')[0] for f in os.listdir(caption_dir) if f.endswith('.json')}
        images_to_process = [img for img in all_images if img.split('.')[0] not in existing_captions]
        
        if images_to_process:
            print(f"📊 Found {len(existing_captions)} existing captions, {len(images_to_process)} new to process")
        else:
            print("✅ All images already have captions!")
    
    # Process images
    if images_to_process:
        if args.parallel:
            print(f"🚀 Using parallel processing with {args.workers} workers")
            results = map_captions_parallel(images_to_process, data_dir, max_workers=args.workers)
        else:
            print("🔄 Using sequential processing")
            map_captions(images_to_process, root=data_dir)
    
    # Generate demo
    print("\n🎨 Generating demo...")
    if args.num_demos > 1:
        make_multiple_demos(data_dir=data_dir, caption_dir=caption_dir, num_demos=args.num_demos)
    else:
        make_demo(data_dir=data_dir, caption_dir=caption_dir)
    
    # Show cache stats if using cache
    if args.use_cache:
        cache = CaptionCache()
        stats = cache.stats()
        print(f"\n💾 Cache stats: {stats['cached_items']} items, {stats['total_size_mb']:.1f}MB")
    
    print("✅ Processing complete!")

if __name__ == "__main__":
    main()
