# Agri Vision — Crop Disease Detection

A Streamlit wrapper for the trained MobileNetV3-Small crop-disease classifier.

## Run it

1. Keep the model files in either of these locations:
   - `best.pt` in this project folder (the current layout), or
   - `model/best.pt` (recommended for deployment).
2. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Start the app:

   ```powershell
   streamlit run app.py
   ```

The app opens in your browser. Upload a JPG, JPEG, PNG, or WebP leaf photo and select **Predict disease**.

## Notes

- The model is loaded once and reused during the app session.
- It uses the same 128×128 resizing and ImageNet normalization used by `train.py`.
- Its predictions cover the 38 trained crop/disease classes saved inside `best.pt`.
