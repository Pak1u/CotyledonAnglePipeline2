import os
import glob
import json
import asyncio
import time
import random
from sam_slicer import segment_and_save_crops
from measurer import analyze_sapling

# Configuration
INPUT_DIR = "Images"
OUTPUT_DIR = "Output"
MAX_CONCURRENT_WORKERS = 1  # Set to 1 for debugging/unstable APIs
MAX_RETRIES = 5

async def measurement_worker(name, queue, results_list):
    """
    Consumer: Pulls tasks and handles retries for server-side exceptions.
    """
    while True:
        task_data = await queue.get()
        if task_data is None:
            queue.task_done()
            break
            
        img_name, crop_path, retry_count = task_data
        
        try:
            # This call will now raise an Exception for 503s
            result = await asyncio.to_thread(analyze_sapling, crop_path)
            
            if result == "NON_PLANT":
                print(f"      [-] Worker {name}: Dropped noise crop ({os.path.basename(crop_path)})")
            elif result is not None:
                results_list.append({
                    "parent_image": img_name,
                    "crop": os.path.basename(crop_path),
                    "angle": result
                })
                print(f"      [✓] Worker {name} Measured: {result}°")
                
        except Exception as e:
            # CATCH server-side exceptions here for re-queuing
            if retry_count < MAX_RETRIES:
                wait_time = (2 ** retry_count) + random.uniform(1, 2)
                print(f"      [!] Worker {name}: API Busy ({e}). Re-queueing in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                await queue.put((img_name, crop_path, retry_count + 1))
            else:
                print(f"      [X] Worker {name}: Max retries reached for {os.path.basename(crop_path)}.")
        
        await asyncio.sleep(0.5)
        queue.task_done()

async def main_event_loop():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    images = glob.glob(os.path.join(INPUT_DIR, "*.*"))
    task_queue = asyncio.Queue()
    all_results = []

    print(f"--- Starting Event-Driven Pipeline for {len(images)} images ---")

    workers = [asyncio.create_task(measurement_worker(f"B-{i}", task_queue, all_results)) 
               for i in range(MAX_CONCURRENT_WORKERS)]

    for img_path in images:
        img_name = os.path.basename(img_path)
        print(f"\n>>> PRODUCER: Slicing {img_name}")
        
        subfolder = os.path.join(OUTPUT_DIR, img_name.split('.')[0])
        crops = segment_and_save_crops(img_path, subfolder)
        
        if crops:
            for crop in crops:
                await task_queue.put((img_name, crop, 0))

    await task_queue.join()

    for _ in range(MAX_CONCURRENT_WORKERS):
        await task_queue.put(None)
    await asyncio.gather(*workers)

    # Compile Report
    final_report = {}
    for res in all_results:
        parent = res["parent_image"]
        if parent not in final_report: final_report[parent] = []
        final_report[parent].append({"crop": res["crop"], "angle": res["angle"]})

    with open(os.path.join(OUTPUT_DIR, "final_event_report.json"), "w") as f:
        json.dump(final_report, f, indent=4)
    
    print("\nProcessing Complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main_event_loop())
    except KeyboardInterrupt:
        print("\n[!] Process interrupted.")