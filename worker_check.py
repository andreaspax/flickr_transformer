import time
import dataset
import torch
import torch.multiprocessing as mp
import tqdm

def slow_collate(batch):
    time.sleep(0.1)  # Simulate slow processing
    return dataset._collate_fn(batch)

if __name__ == '__main__':
    # Set multiprocessing context
    mp.set_start_method('spawn', force=True)
    
    # Create test dataset
    # train_ds = dataset.FlickrClipDataset(dataset.train_ds)
    test_ds = dataset.FlickrClipDataset(dataset.test_ds)
    # Test with different worker counts
    for workers in [0, 2, 4]:
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=126,
            collate_fn=slow_collate,
            num_workers=workers,
            persistent_workers=workers > 0
        )
        
        start = time.time()
        for batch in tqdm.tqdm(test_loader, desc="Processing batches", unit="batch", leave=False):
            pass
            
        print(f"Workers: {workers}, Time: {time.time()-start:.2f}s")