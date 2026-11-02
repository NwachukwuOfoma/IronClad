# IronClad

A face image retrieval system, built as a Johns Hopkins course project (705.603).

## What it does

- Extracts face embeddings with FaceNet (InceptionResnetV1) using VGGFace2 or CASIA-WebFace pretrained weights
- Provides FAISS brute force, HNSW, and LSH index classes with cosine, Euclidean, and dot product measures
- Serves search through a Flask API
- Includes an evaluation script that measures MRR and top N accuracy, and tests robustness to noise, resizing, and brightness changes

## Running it

```bash
cd ironclad
pip install -r requirements.txt
python app.py
```

The face image dataset is not included, so you need to supply your own gallery and probe images.

## Notes

Parts of this project build on starter code provided for the course. Course data, tests, and written reports are not included in this repository.
