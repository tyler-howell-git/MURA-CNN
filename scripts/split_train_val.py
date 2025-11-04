import pandas as pd
from sklearn.model_selection import train_test_split

def split_save(csv_in, train_out, val_out, test_size=0.2, seed=42): #80 train / 20 val
    df = pd.read_csv(csv_in, header=None)
    df.columns = ['study_path', 'label']
    
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['label'],
        random_state=seed
    )
    
    train_df.to_csv(train_out, index=False, header=False)
    val_df.to_csv(val_out, index=False, header=False)
    print(f"Saved {len(train_df)} train and {len(val_df)} val samples")

if __name__ == "__main__":
    split_save(
        csv_in="data/splits/train_labeled_studies.csv",
        train_out="data/splits/train_labeled_studies_split.csv",
        val_out="data/splits/val_labeled_studies_split.csv"
    )
