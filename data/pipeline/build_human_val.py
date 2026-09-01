import os
import pandas as pd
DATA = os.environ.get('ECUP_DATA', 'data')
VAL_FOLD = 0

def main():
    folds = pd.read_parquet(f'{DATA}/artifacts/folds_v1.parquet')
    val = folds[folds.fold == VAL_FOLD].reset_index(drop=True)
    out = val[['id1', 'id2', 'target', 'category']].rename(columns={'target': 'label'})
    out.to_parquet(f'{DATA}/artifacts/humanval_fold0.parquet', index=False)
    print(f'валидация: {len(out):,} пар, позитивов {out.label.mean() * 100:.2f}%')

if __name__ == '__main__':
    main()
