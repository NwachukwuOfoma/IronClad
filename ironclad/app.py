import os.path

import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify

from modules.extraction.embedding import Embedding
from modules.extraction.preprocessing import Preprocessing
from modules.retrieval.index.hnsw import FaissHNSW
from modules.retrieval.search import FaissSearch

app = Flask(__name__)

## List of designed parameters: 
# (Configure these parameters according to your design decisions)
DEFAULT_K = '3'
MODEL = 'vggface2'
INDEX = 'hnsw' # This should be 'hnsw', 'lsh', or 'bruteforce' based on your choice
SIMILARITY_MEASURE = 'cosine'
SAVE_INDEX = False
INDEX_PATH = 'scripts/faiss_hnsw_index.pkl' # This path should be updated based on INDEX choice
# Add more if needed...
preprocessor = Preprocessing()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Embedding(pretrained=MODEL, device=device)
index = None

# Initialize the correct index type based on the INDEX variable
if INDEX == 'hnsw':
    index_class = FaissHNSW
elif INDEX == 'lsh':
    from modules.retrieval.index.lsh import FaissLSH
    index_class = FaissLSH
elif INDEX == 'bruteforce':
    from modules.retrieval.index.bruteforce import FaissBruteForce
    index_class = FaissBruteForce
else:
    raise ValueError("Unsupported INDEX type. Choose 'hnsw', 'lsh', or 'bruteforce'.")

if SAVE_INDEX and os.path.exists(INDEX_PATH):
    index = index_class.load(INDEX_PATH) # Load data from index
else:
    # Ensure dim is consistent with the model's output (FaceNet outputs 512-dim embeddings)
    index = index_class(dim=512, metric=SIMILARITY_MEASURE)
    
search = FaissSearch(index, metric=SIMILARITY_MEASURE)

@app.route('/identify', methods=['POST'])
def identify():
    """
    Process the probe image to identify top-k identities in the gallery.

    Expects form-data with:
      - image: Image file to be processed.
      - k: (optional) Integer specifying the number of top identities 
           (default is 3).

    Returns:
      JSON response with a success message and the provided value of k.
      If errors occur, returns a JSON error message with the appropriate status code.
    """
    # Check if the request has the image file
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    # Retrieve and validate the integer parameter "k"
    try:
        k = int(request.form.get('k', DEFAULT_K))
    except ValueError:
        return jsonify({"error": "Invalid integer for parameter 'k'"}), 400

    # Convert the image into a NumPy array
    try:
        image = np.array(Image.open(file))
        print(image)
    except Exception as e:
        return jsonify({
            "error": "Failed to convert image to numpy array",
            "details": str(e)
        }), 500

    image_tensor = preprocessor.process(Image.fromarray(image))
    embedding = model.encode(image_tensor)
    distances, indices, metadatas = search.search(embedding, k)
    
    # Extract identities, handling cases where metadata might be None or empty
    # The search method should return a list of lists for metadatas,
    # where each inner list corresponds to the k results for a given query.
    # Since we are searching with a single query (1 probe image), metadatas[0] gives us its results.
    # Then we filter out any None values.
    identities = [name for name in metadatas[0] if name is not None]

    return jsonify({
        "message": f"Returned top-{k} identities",
        "ranked identities": identities
    }), 200


@app.route("/add", methods=['POST'])
def add():
    """
    Add a provided image to the gallery with an associated name.

    Expects form-data with:
      - image: Image file to be added.
      - name: String representing the identity associated with the image.

    Returns:
      JSON response confirming the image addition.
      If errors occur, returns a JSON error message with the appropriate status code.
    """
    # Check if the request has the image file
    if 'image' not in request.files:
        return jsonify({"Error": "No image part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"Error": "No file selected for uploading"}), 400

    # Convert the image into a NumPy array
    try:
        image = np.array(Image.open(file))
    except Exception as e:
        return jsonify({
            "Error": "Failed to convert image to numpy array", # Changed from 'error' to 'Error' for consistency
            "details": str(e)
        }), 500

    # Retrieve the 'name' parameter
    name = request.form.get('name')
    if not name:
        return jsonify({"Error": "Must have associated 'name'"}), 400

    # Check for duplicate name before adding
    if name in index.metadata:
        return jsonify({"Error": f"The image for identity '{name}' already exists in the catalog."}), 400

    image_tensor = preprocessor.process(Image.fromarray(image))
    embedding = model.encode(image_tensor)
    index.add_embeddings([embedding], [name])

    if SAVE_INDEX:
        # Ensure the directory for INDEX_PATH exists if you're saving the index
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        index.save(INDEX_PATH)

    return jsonify({
        "message": f"New image added to gallery (as {name}) and indexed into catalog."
    }), 200 # Added 200 status code for success, although Flask defaults to 200.


if __name__ == '__main__':
    # Add the MacOS specific environment variable for HNSW if needed
    # import os
    # os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 
    app.run(port=5000, debug=True, host='0.0.0.0')