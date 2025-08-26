"""
Simple caching system for image captions
"""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict


class CaptionCache:
    """Smart caching system to avoid reprocessing"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash of file for cache key"""
        stat = os.stat(file_path)
        # Use size and modification time for quick hash
        return hashlib.md5(f"{file_path}_{stat.st_size}_{stat.st_mtime}".encode()).hexdigest()
    
    def get(self, image_path: str) -> Optional[Dict]:
        """Retrieve cached caption if valid"""
        try:
            file_hash = self._get_file_hash(image_path)
            cache_file = self.cache_dir / f"{file_hash}.json"
            
            if cache_file.exists():
                # Check if cache is still valid (e.g., less than 30 days old)
                cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if cache_age < timedelta(days=30):
                    with open(cache_file, "r") as f:
                        return json.load(f)
        except Exception as e:
            print(f"Cache read error: {e}")
        return None
    
    def set(self, image_path: str, caption: Dict):
        """Store caption in cache"""
        try:
            file_hash = self._get_file_hash(image_path)
            cache_file = self.cache_dir / f"{file_hash}.json"
            
            with open(cache_file, "w") as f:
                json.dump(caption, f, indent=2)
            
            # Update metadata
            self.metadata[image_path] = {
                "hash": file_hash,
                "cached_at": datetime.now().isoformat()
            }
            self._save_metadata()
        except Exception as e:
            print(f"Cache write error: {e}")
    
    def clear(self):
        """Clear all cache"""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        self.metadata = {}
        self._save_metadata()
        print("✅ Cache cleared")
    
    def stats(self) -> Dict:
        """Get cache statistics"""
        cache_files = list(self.cache_dir.glob("*.json"))
        cache_files = [f for f in cache_files if f.name != "metadata.json"]
        
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "cached_items": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir)
        }
