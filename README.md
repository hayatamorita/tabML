# Tabular ML Template

表形式データ向けのML実験テンプレートです。`notebooks/main.ipynb`でCONFIGを編集し、共通処理は`src/*.py`から呼び出します。

## Structure

```text
tabular-ml/
├── README.md
├── requirements.txt
├── notebooks/main.ipynb
├── data/raw/
├── src/
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── metrics.py
│   └── utils.py
└── outputs/
```

## Setup

```bash
pip install -r requirements.txt
```

Optional:

```bash
pip install autogluon.tabular
pip install tabpfn
```

## Usage

1. `data/raw/train.csv`を置きます。
2. 必要なら`data/raw/test.csv`も置きます。
3. `notebooks/main.ipynb`のCONFIGを編集します。
4. notebookを上から実行します。

`train.csv`にはCONFIGの`data.target_col`で指定した目的変数列が必要です。`test.csv`は任意で、目的変数列が含まれていても予測時には無視されます。

## Models

`CONFIG["models"]`に複数モデルを並べると一括比較できます。`enabled=False`のモデルはskipされます。同じモデルを複数条件で回す場合は`alias`を変えてください。

## CV / Holdout

`CONFIG["split"]["cv"] = True`ならCV、`False`ならholdout validationです。回帰でstratifiedを使う場合は目的変数を分位点binningして疑似stratifiedにします。

## Outputs

実行ごとに`outputs/YYYYMMDD-rand4/`が作成されます。

- `<model>/models/`: model/preprocessor pickle
- `<model>/predictions/`: OOF、valid、test、fold metrics
- `<model>/reports/`: metrics、model_config
- `summary/`: `config.json`、`model_comparison.csv`

既存run directoryは上書きせず、新しいrun directoryを作成します。
