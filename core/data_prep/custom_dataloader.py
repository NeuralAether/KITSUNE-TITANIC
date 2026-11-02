import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List, Optional, Dict, Any


class TitanicDataset(Dataset):
    """
    High-performance PyTorch Dataset for tabular + text data.
    Eliminates pandas indexing overhead in __getitem__ by using
    pre-extracted PyTorch tensors and native Python lists.
    """
    def __init__(
        self,
        path: str,
        target_col: Optional[str] = 'Survived',
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        droppable_cols: Optional[List[str]] = None,
        id_col: Optional[str] = 'PassengerId',
        is_binary_target: bool = True, 
        is_train: bool = True
    ):
        self.df = pd.read_csv(path)
        self.length = len(self.df)
        
        # 1. Resolve columns (filter out any columns not present in self.df)
        self.numeric_cols = [c for c in numeric_cols if c in self.df.columns] if numeric_cols else []
        self.categorical_cols = [c for c in categorical_cols if c in self.df.columns] if categorical_cols else []
        self.droppable_cols = [c for c in droppable_cols if c in self.df.columns] if droppable_cols else []
        
        # 2. Pre-convert numeric columns into a single contiguous 2D float32 tensor
        if self.numeric_cols:
            num_data = self.df[self.numeric_cols].to_numpy(dtype='float32')
            self.numeric_tensor = torch.from_numpy(num_data)
        else:
            self.numeric_tensor = None

        # 3. Pre-extract text into native Python lists (avoids slow self.df.iloc during batching)
        self.text_data = {
            col: self.df[col].fillna('').astype(str).tolist() 
            for col in self.categorical_cols
        } if self.categorical_cols else {}

        # 4. Pre-extract IDs as a native list
        self.ids = self.df[id_col].tolist() if id_col and id_col in self.df.columns else None

        # 5. Pre-convert Target (only if this is training data)
        if is_train and target_col and target_col in self.df.columns:
            target_data = self.df[target_col].to_numpy()
            if is_binary_target:
                # Float32 shaped (N, 1) for BCEWithLogitsLoss
                self.target_tensor = torch.from_numpy(target_data).float().unsqueeze(1)
            else:
                # Long shaped (N,) for nn.CrossEntropyLoss
                self.target_tensor = torch.from_numpy(target_data).long()
        else:
            self.target_tensor = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {}

        if self.numeric_tensor is not None:
            item['numeric'] = self.numeric_tensor[idx]

        if self.text_data:
            item['text'] = {col: self.text_data[col][idx] for col in self.categorical_cols}

        if self.ids is not None:
            item['id'] = self.ids[idx]

        if self.target_tensor is not None:
            item['target'] = self.target_tensor[idx]

        return item