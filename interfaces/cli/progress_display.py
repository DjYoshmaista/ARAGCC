"""
Progress display utilities for the CLI with dual progress bars.
"""

import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime
import logging

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

logger = logging.getLogger(__name__)

class DualProgressDisplay:
    """
    Dual progress bar display for files/chunks and embeddings
    """
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.files_pbar = None
        self.embeddings_pbar = None
        self.active = False
        self.last_update = time.time()
        
        # Progress state
        self.files_stats = {'total': 0, 'processed': 0, 'failed': 0}
        self.chunks_stats = {'total': 0, 'processed': 0}  
        self.embeddings_stats = {'total_needed': 0, 'generated': 0, 'stored': 0, 'failed': 0}
        
        # Performance tracking
        self.start_time = time.time()
        self.log_interval = 5.0  # Log every 5 seconds
        self.last_log_time = time.time()
    
    def start(self):
        """Start the progress display"""
        self.active = True
        self.start_time = time.time()
        print(f"🚀 Started processing task: {self.task_id}")
        print("📊 Initializing progress tracking...")
        logger.info(f"Progress tracking started for task {self.task_id}")
    
    def stop(self):
        """Stop the progress display and clean up"""
        self.active = False
        
        if self.files_pbar:
            self.files_pbar.close()
            self.files_pbar = None
        
        if self.embeddings_pbar:
            self.embeddings_pbar.close()  
            self.embeddings_pbar = None
        
        # Clear progress bar lines
        print("\n" * 2)
        
        elapsed = time.time() - self.start_time
        print(f"✅ Task completed in {elapsed:.1f} seconds")
        logger.info(f"Progress tracking stopped for task {self.task_id}")
    
    def update(self, stats: Dict[str, Any]):
        """Update progress bars with new statistics"""
        if not self.active:
            return
        
        try:
            # Extract statistics
            files_data = stats.get('files', {})
            chunks_data = stats.get('chunks', {}) 
            embeddings_data = stats.get('embeddings', {})
            database_data = stats.get('database', {})
            performance_data = stats.get('performance', {})
            
            # Update internal state
            self.files_stats = files_data
            self.chunks_stats = chunks_data
            self.embeddings_stats = embeddings_data
            
            # Initialize or update files/chunks progress bar
            self._update_files_progress_bar(files_data, chunks_data)
            
            # Initialize or update embeddings progress bar
            self._update_embeddings_progress_bar(embeddings_data)
            
            # Log detailed progress periodically
            current_time = time.time()
            if current_time - self.last_log_time >= self.log_interval:
                self._log_detailed_progress(stats)
                self.last_log_time = current_time
            
            self.last_update = current_time
            
        except Exception as e:
            logger.error(f"Error updating progress display: {e}")
    
    def _update_files_progress_bar(self, files_data: Dict[str, Any], chunks_data: Dict[str, Any]):
        """Update the files/chunks progress bar"""
        if not TQDM_AVAILABLE:
            self._show_text_progress("Files", files_data)
            return
        
        total_files = files_data.get('total', 0)
        processed_files = files_data.get('processed', 0)
        failed_files = files_data.get('failed', 0)
        total_chunks = chunks_data.get('total', 0)
        processed_chunks = chunks_data.get('processed', 0)
        
        # Initialize progress bar if needed
        if self.files_pbar is None and total_files > 0:
            self.files_pbar = tqdm(
                total=total_files,
                desc="📁 Files/Chunks",
                unit="files",
                position=0,
                leave=True,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]",
                dynamic_ncols=True
            )
            logger.info("Files progress bar initialized")
        
        # Update progress bar
        if self.files_pbar and total_files > 0:
            current_progress = processed_files + failed_files
            self.files_pbar.n = min(current_progress, total_files)
            
            # Update postfix with detailed info
            postfix = {
                'processed': processed_files,
                'chunks': f"{processed_chunks}/{total_chunks}" if total_chunks > 0 else "0",
                'failed': failed_files if failed_files > 0 else ""
            }
            
            # Remove empty values from postfix
            postfix = {k: v for k, v in postfix.items() if v != ""}
            
            self.files_pbar.set_postfix(postfix)
            self.files_pbar.refresh()
    
    def _update_embeddings_progress_bar(self, embeddings_data: Dict[str, Any]):
        """Update the embeddings progress bar"""
        if not TQDM_AVAILABLE:
            self._show_text_progress("Embeddings", embeddings_data)
            return
        
        total_needed = embeddings_data.get('total_needed', 0)
        generated = embeddings_data.get('generated', 0)
        stored = embeddings_data.get('stored', 0)
        failed = embeddings_data.get('failed', 0)
        
        # Initialize progress bar if needed
        if self.embeddings_pbar is None and total_needed > 0:
            self.embeddings_pbar = tqdm(
                total=total_needed,
                desc="🔮 Embeddings", 
                unit="emb",
                position=1,
                leave=True,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]",
                dynamic_ncols=True
            )
            logger.info("Embeddings progress bar initialized")
        
        # Update progress bar
        if self.embeddings_pbar and total_needed > 0:
            self.embeddings_pbar.n = min(generated, total_needed)
            
            # Update postfix with detailed info
            postfix = {
                'stored': stored,
                'failed': failed if failed > 0 else "",
                'pending': max(0, total_needed - generated)
            }
            
            # Remove empty values from postfix
            postfix = {k: v for k, v in postfix.items() if v != ""}
            
            self.embeddings_pbar.set_postfix(postfix)
            self.embeddings_pbar.refresh()
    
    def _show_text_progress(self, category: str, data: Dict[str, Any]):
        """Show text-based progress when tqdm is not available"""
        if category == "Files":
            total = data.get('total', 0)
            processed = data.get('processed', 0)
            failed = data.get('failed', 0)
            percent = data.get('progress_percent', 0)
            
            if total > 0:
                print(f"📁 Files: {processed}/{total} ({percent:.1f}%) - Failed: {failed}")
        
        elif category == "Embeddings":
            total = data.get('total_needed', 0)
            generated = data.get('generated', 0)
            stored = data.get('stored', 0)
            failed = data.get('failed', 0)
            percent = data.get('progress_percent', 0)
            
            if total > 0:
                print(f"🔮 Embeddings: {generated}/{total} ({percent:.1f}%) - Stored: {stored}, Failed: {failed}")
    
    def _log_detailed_progress(self, stats: Dict[str, Any]):
        """Log detailed progress information"""
        try:
            files_data = stats.get('files', {})
            chunks_data = stats.get('chunks', {})
            embeddings_data = stats.get('embeddings', {})
            database_data = stats.get('database', {})
            performance_data = stats.get('performance', {})
            
            elapsed = time.time() - self.start_time
            
            # Log file progress
            logger.info(
                f"Progress Update - Files: {files_data.get('processed', 0)}/{files_data.get('total', 0)} "
                f"({files_data.get('progress_percent', 0):.1f}%), "
                f"Chunks: {chunks_data.get('processed', 0)}/{chunks_data.get('total', 0)}"
            )
            
            # Log embedding progress
            logger.info(
                f"Progress Update - Embeddings: {embeddings_data.get('generated', 0)}/{embeddings_data.get('total_needed', 0)} "
                f"({embeddings_data.get('progress_percent', 0):.1f}%), "
                f"Stored: {embeddings_data.get('stored', 0)}, "
                f"Failed: {embeddings_data.get('failed', 0)}"
            )
            
            # Log performance metrics
            files_per_sec = performance_data.get('files_per_second', 0)
            embeddings_per_sec = performance_data.get('embeddings_per_second', 0)
            
            logger.info(
                f"Performance - Files/sec: {files_per_sec:.2f}, "
                f"Embeddings/sec: {embeddings_per_sec:.2f}, "
                f"Elapsed: {elapsed:.1f}s"
            )
            
            # Log database statistics
            docs_stored = database_data.get('documents_stored', 0)
            chunks_stored = database_data.get('chunks_stored', 0)
            vectors_stored = database_data.get('vectors_stored', 0)
            
            if docs_stored > 0 or chunks_stored > 0 or vectors_stored > 0:
                logger.info(
                    f"Database - Documents: {docs_stored}, "
                    f"Chunks: {chunks_stored}, "
                    f"Vectors: {vectors_stored}"
                )
                
        except Exception as e:
            logger.error(f"Error logging progress details: {e}")
    
    def show_final_summary(self, stats: Dict[str, Any]):
        """Show final summary of processing"""
        try:
            files_data = stats.get('files', {})
            chunks_data = stats.get('chunks', {})
            embeddings_data = stats.get('embeddings', {})
            database_data = stats.get('database', {})
            performance_data = stats.get('performance', {})
            
            elapsed = time.time() - self.start_time
            
            print("\n" + "="*60)
            print("📊 PROCESSING SUMMARY")
            print("="*60)
            
            # Files summary
            print(f"📁 Files Processed: {files_data.get('processed', 0)}/{files_data.get('total', 0)}")
            if files_data.get('failed', 0) > 0:
                print(f"   ❌ Failed: {files_data.get('failed', 0)}")
            
            # Chunks summary
            print(f"📄 Chunks Created: {chunks_data.get('total', 0)}")
            print(f"    Chunks Processed: {chunks_data.get('processed', 0)}")
            
            # Embeddings summary
            print(f"🔮 Embeddings Generated: {embeddings_data.get('generated', 0)}/{embeddings_data.get('total_needed', 0)}")
            print(f"    Embeddings Stored: {embeddings_data.get('stored', 0)}")
            if embeddings_data.get('failed', 0) > 0:
                print(f"    ❌ Failed: {embeddings_data.get('failed', 0)}")
            
            # Database summary
            print(f"💾 Database Storage:")
            print(f"    Documents: {database_data.get('documents_stored', 0)}")
            print(f"    Chunks: {database_data.get('chunks_stored', 0)}")  
            print(f"    Vectors: {database_data.get('vectors_stored', 0)}")
            
            # Performance summary
            files_per_sec = performance_data.get('files_per_second', 0)
            embeddings_per_sec = performance_data.get('embeddings_per_second', 0)
            
            print(f"⚡ Performance:")
            print(f"    Total Time: {elapsed:.1f} seconds")
            print(f"    Files/sec: {files_per_sec:.2f}")
            print(f"    Embeddings/sec: {embeddings_per_sec:.2f}")
            
            print("="*60)
            
            # Log summary as well
            logger.info(f"Processing Summary - Task {self.task_id} completed in {elapsed:.1f}s")
            logger.info(f"Files: {files_data.get('processed', 0)}/{files_data.get('total', 0)}, "
                       f"Embeddings: {embeddings_data.get('generated', 0)}/{embeddings_data.get('total_needed', 0)}")
            
        except Exception as e:
            logger.error(f"Error showing final summary: {e}")
            print(f"\n✅ Task {self.task_id} completed (summary unavailable)")

def install_tqdm_if_needed():
    """Helper to suggest tqdm installation"""
    if not TQDM_AVAILABLE:
        print("📊 Note: Install 'tqdm' for enhanced progress bars: pip install tqdm")
        print("    Using text-based progress display for now.")
        return False
    return True