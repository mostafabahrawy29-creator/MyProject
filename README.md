# AutoValue — Streamlit Car Price Predictor

## Files
- `app.py`: Streamlit application
- `requirements.txt`: Python dependencies
- `car details v4 (1).csv`: your dataset (place this beside `app.py`)

## Run locally
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The app follows the preprocessing and LinearRegression pipeline in the supplied
`Cars.ipynb`. It uses `np.log(Price)` for training and `np.exp(prediction)` to
convert the prediction back to the original price scale.
