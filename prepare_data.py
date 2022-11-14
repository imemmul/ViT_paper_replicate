import os
from xmlrpc.client import Boolean

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str=None, 
    test_dir: str=None,
    test_data=None,
    train_data=None,
    data_folder_imported: Boolean=False,
    transform=None, 
    batch_size: int=32, 
    num_workers: int=NUM_WORKERS):

  train_dataloader, test_dataloader = 0, 0
  train_data, test_data = train_data, test_data
  # Use ImageFolder to create dataset(s)
  if data_folder_imported:
    train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True
    )
    test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True
    )
  else:
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
    )
    test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
    )

  # Get class names
  class_names = None
  try:
    class_names = train_data.classes
  except:
    pass

  return train_dataloader, test_dataloader, class_names
