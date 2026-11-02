# {ROOT_DIR}/ironclad/evaluation.py

import os
import tarfile
from collections import defaultdict
from PIL import Image, ImageEnhance
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm # For progress bars
import math # For math.isclose in floating point comparisons

# Import your existing modules
from modules.extraction.embedding import Embedding
from modules.extraction.preprocessing import Preprocessing
from modules.retrieval.index.bruteforce import FaissBruteForce # Only brute-force for this assignment
from modules.retrieval.search import FaissSearch

# --- Configuration Parameters ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Adjust if your root is different
STORAGE_DIR = os.path.join(ROOT_DIR, 'storage')
DATA_ARCHIVE = os.path.join(ROOT_DIR, 'multi_image_identities.tar') # Path to your tar file
OUTPUT_MD_FILE = os.path.join(os.path.dirname(__file__), 'CaseStudy.md')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures') # Directory for your generated plots

# Define models to compare
MODELS = {
    'casia-webface': 'casia-webface',
    'vggface2': 'vggface2'
}

# Define metrics to use for search (Euclidean, Cosine, Dot Product)
SEARCH_METRICS = ['cosine', 'euclidean', 'dot_product'] 

# Embeddings dimension (FaceNet outputs 512-dim)
EMBEDDING_DIM = 512

# Device for model inference
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Markdown Logging Helper ---

def append_to_markdown(file_handle, content):
    """Appends content to the markdown file."""
    file_handle.write(content + "\n")

def create_markdown_table_str(headers, data, title=None, caption=None):
    """Generates a Markdown table string from data."""
    md_str = ""
    if title:
        md_str += f"### {title}\n\n"
    
    if not data:
        md_str += "No data to display in table.\n\n"
        return md_str

    # Table Header
    md_str += "| " + " | ".join(headers) + " |\n"
    # Table Separator
    md_str += "|" + "---|".join(["-" * len(h) for h in headers]) + "|\n"

    # Data Rows
    for row_data in data:
        row_values = []
        for header in headers:
            value = row_data.get(header, '')
            if isinstance(value, float):
                row_values.append(f"{value:.4f}") # Format floats
            else:
                row_values.append(str(value))
        md_str += "| " + " | ".join(row_values) + " |\n"
    
    if caption:
        md_str += f"\n***{caption}***\n\n"
    return md_str


# --- 1. Data Preparation Functions ---

def extract_and_organize_data(archive_path, extract_to_path):
    """
    Extracts the tar archive and organizes images into gallery and probe dictionaries.
    Returns:
        gallery_data (dict): {identity_name: [list of image paths]}
        probe_data (dict): {identity_name: [list of image paths]}
    """
    if not os.path.exists(extract_to_path):
        os.makedirs(extract_to_path)
    
    # Check if data is already extracted to avoid re-extracting every time
    if os.path.exists(os.path.join(extract_to_path, 'gallery')) and \
       os.path.exists(os.path.join(extract_to_path, 'probe')):
        print(f"Data already extracted to {extract_to_path}. Skipping extraction.")
    else:
        print(f"Extracting data from {archive_path} to {extract_to_path}...")
        with tarfile.open(archive_path, 'r') as tar:
            tar.extractall(path=extract_to_path)
        print("Extraction complete.")

    gallery_path = os.path.join(extract_to_path, 'multi_image_gallery') 
    probe_path = os.path.join(extract_to_path, 'probe')

    gallery_data = defaultdict(list)
    probe_data = defaultdict(list)

    # Populate gallery_data
    if os.path.exists(gallery_path):
        for identity_name in os.listdir(gallery_path):
            identity_dir = os.path.join(gallery_path, identity_name)
            if os.path.isdir(identity_dir):
                for img_name in os.listdir(identity_dir):
                    # --- ADD THIS CHECK ---
                    if img_name.startswith('._'):
                        continue # Skip macOS metadata files
                    # --- END ADDITION ---
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        gallery_data[identity_name].append(os.path.join(identity_dir, img_name))
    
    # Populate probe_data
    if os.path.exists(probe_path):
        for identity_name in os.listdir(probe_path):
            identity_dir = os.path.join(probe_path, identity_name)
            if os.path.isdir(identity_dir):
                for img_name in os.listdir(identity_dir):
                    # --- ADD THIS CHECK ---
                    if img_name.startswith('._'):
                        continue # Skip macOS metadata files
                    # --- END ADDITION ---
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        probe_data[identity_name].append(os.path.join(identity_dir, img_name))

    print(f"Loaded {len(gallery_data)} gallery identities and {len(probe_data)} probe identities.")
    return gallery_data, probe_data


# --- 2. Augmentation Functions ---

class AugmentationPreprocessor(Preprocessing):
    """
    Extends the base Preprocessing class to include image augmentations.
    """
    def __init__(self, image_size: int = 160):
        super().__init__(image_size)
        # Store default transform
        self.default_transform = self.preprocess_transform

    def apply_gaussian_noise(self, image: Image.Image, mean: float = 0.0, std: float = 0.1) -> Image.Image:
        """Applies Gaussian noise to a PIL Image."""
        img_np = np.array(image).astype(np.float32) / 255.0
        noise = np.random.normal(mean, std, img_np.shape).astype(np.float32)
        noisy_img_np = img_np + noise
        noisy_img_np = np.clip(noisy_img_np, 0.0, 1.0)
        return Image.fromarray((noisy_img_np * 255).astype(np.uint8))

    def apply_resize(self, image: Image.Image, size_factor: float = 1.0) -> Image.Image:
        """Resizes a PIL Image by a factor relative to its original size."""
        original_width, original_height = image.size
        new_width = int(original_width * size_factor)
        new_height = int(original_height * size_factor)
        
        # Ensure new dimensions are at least 1x1 to avoid errors
        new_width = max(1, new_width)
        new_height = max(1, new_height)

        # Use Image.Resampling.LANCZOS for high-quality downsampling/upsampling
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


    def apply_brightness(self, image: Image.Image, factor: float = 1.0) -> Image.Image:
        """Adjusts brightness of a PIL Image."""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def process_with_augmentation(self, probe: Image.Image, augmentation_type: str = None, **kwargs) -> torch.Tensor:
        """
        Processes an image with an optional augmentation applied before standard preprocessing.
        """
        augmented_image = probe

        if augmentation_type == 'gaussian_noise':
            augmented_image = self.apply_gaussian_noise(probe, **kwargs)
        elif augmentation_type == 'resize':
            augmented_image = self.apply_resize(probe, **kwargs)
        elif augmentation_type == 'brightness':
            augmented_image = self.apply_brightness(probe, **kwargs)
        elif augmentation_type is not None: # Catch-all for unknown augmentation types
            raise ValueError(f"Unknown augmentation type: {augmentation_type}")
        
        # Apply the standard preprocessing transform after augmentation
        return self.default_transform(augmented_image).unsqueeze(0).to(self.device)


# --- 3. Evaluation Metrics Functions ---

def calculate_metrics(ranked_results, ground_truth_identity, k_values):
    """
    Calculates various evaluation metrics for a single query.
    ranked_results: List of identities in order of relevance returned by search.
    ground_truth_identity: The true identity of the probe.
    k_values: List of k values for which to calculate precision/recall/accuracy.
    Returns:
        metrics (dict): Dictionary of calculated metrics.
    """
    metrics = {}
    
    # Mean Reciprocal Rank (MRR)
    try:
        # Find the rank of the first correct answer
        rank = -1
        for i, result_id in enumerate(ranked_results):
            if result_id == ground_truth_identity:
                rank = i + 1
                break
        metrics['mrr'] = 1.0 / rank if rank != -1 else 0.0
    except ValueError: # Should not happen with current logic, but as safeguard
        metrics['mrr'] = 0.0 

    # Top-N Accuracy (for each k in k_values)
    for k in k_values:
        top_k_results = ranked_results[:k]
        metrics[f'top_{k}_accuracy'] = 1.0 if ground_truth_identity in top_k_results else 0.0
    
    return metrics


# --- 4. Experiment Orchestration ---

def run_evaluation_experiment(
    model_name: str, 
    search_metric: str, 
    gallery_data: dict, 
    probe_data: dict, 
    embedding_model: Embedding, 
    preprocessor_obj: AugmentationPreprocessor,
    augmentation_type: str = None, 
    augmentation_params: dict = None,
    k_max: int = 20 # Max K for evaluation metrics (covers Task 5 as well)
):
    """
    Runs a complete evaluation experiment for a given model, search metric, and augmentation.
    """
    exp_description = f"Model={model_name}, Metric={search_metric}"
    if augmentation_type:
        params_str = ", ".join([f"{k}={v}" for k, v in augmentation_params.items()])
        exp_description += f", Augmentation={augmentation_type} ({params_str})"
    print(f"\n--- Running Experiment: {exp_description} ---")

    try: # <--- ADD THIS TRY BLOCK HERE

        # 1. Build Gallery Index
        gallery_embeddings = []
        gallery_metadata = []
        print("Building gallery index...")
        # --- Debugging: Print some gallery identity examples --- (keep this if you like)
        print("\n--- Debugging Gallery Identity Examples ---")
        for i, (identity_name, img_paths) in enumerate(gallery_data.items()):
            if i < 5: 
                print(f"Gallery Identity: '{identity_name}', Example Path: '{img_paths[0]}'")
            else:
                break
        print("-------------------------------------------\n")
        # --- END Debugging ---

        for identity_name, img_paths in tqdm(gallery_data.items(), desc="Processing Gallery"):
            for img_path in img_paths:
                img = Image.open(img_path).convert('RGB')
                processed_img = preprocessor_obj.process(img) 
                embedding = embedding_model.encode(processed_img)
                gallery_embeddings.append(embedding)
                gallery_metadata.append(identity_name)
        
        index = FaissBruteForce(dim=EMBEDDING_DIM, metric=search_metric)
        index.add_embeddings(gallery_embeddings, gallery_metadata)
        
        searcher = FaissSearch(index, metric=search_metric)

        # 2. Process Probes and Evaluate
        all_mrr = []
        all_top_k_accuracy = defaultdict(list) 
        
        print("Processing probes and evaluating performance...")
        probe_identities = list(probe_data.keys())
        
        # --- Debugging: Keep track of a few debug probes ---
        debug_probe_count = 0
        max_debug_probes = 20 # Show more debug outputs now
        # --- END Debugging ---

        # --- Pre-calculate gallery identity set for quick lookup ---
        gallery_identity_names_set = set(gallery_data.keys())
        # --- END Pre-calculation ---

        for identity_name in tqdm(probe_identities, desc="Processing Probes"):
            if not probe_data[identity_name]:
                continue 

            # --- NEW DEBUGGING: Check if Ground Truth Identity exists in gallery ---
            is_ground_truth_in_gallery_dataset = identity_name in gallery_identity_names_set
            if not is_ground_truth_in_gallery_dataset:
                if debug_probe_count < max_debug_probes:
                     print(f"\n--- Debugging Probe: '{identity_name}' (SKIPPING, GT NOT IN GALLERY) ---")
                     print(f"  Ground Truth Identity '{identity_name}' not found in gallery dataset. Skipping evaluation for this probe.")
                     print("-------------------------------------------\n")
                continue # Skip this probe if its identity isn't in the gallery

            # ... (rest of processing probe image, embedding, and search) ...

            probe_image_path = probe_data[identity_name][0] 
            probe_image = Image.open(probe_image_path).convert('RGB')
            
            if augmentation_type:
                processed_probe_tensor = preprocessor_obj.process_with_augmentation(
                    probe_image, augmentation_type, **augmentation_params
                )
            else:
                processed_probe_tensor = preprocessor_obj.process(probe_image)
                
            probe_embedding = embedding_model.encode(processed_probe_tensor)

            distances, indices, metadatas = searcher.search(probe_embedding, k=k_max)
            ranked_results = metadatas[0] 
            
            # --- NEW DEBUGGING: Also get true distance to GT in gallery if possible ---
            # Make gallery_embeddings available here for accurate debugging:
            # You might need to pass gallery_embeddings as an argument to this function
            # or make it a global/class member for accurate comparison here.
            # For now, let's simplify this debug check to avoid complicating function signature:
            # We already have min_true_distance_in_gallery calculated in main loop debug print
            # from previous iteration. For now, let's remove the min_true_distance_in_gallery recalculation here
            # to simplify, as it was only for debug print and not for core logic.
            # If you want to keep it, you'd need to pass `gallery_embeddings` to this function.
            
            # --- Debugging: Print for specific probes ---
            if debug_probe_count < max_debug_probes:
                print(f"\n--- Debugging Probe: '{identity_name}' ---")
                print(f"  Probe Image Path: {probe_image_path}")
                print(f"  Ground Truth Identity: '{identity_name}'")
                print(f"  Is Ground Truth in Gallery (by name)? {is_ground_truth_in_gallery_dataset}")
                
                # To print Min Distance to True Match, we need to calculate it outside if not passed
                # For now, just focus on the rank results:
                print(f"  Top {k_max} Ranked Results: {ranked_results}")
                
                if identity_name in ranked_results:
                    match_rank = ranked_results.index(identity_name) + 1
                    match_distance = distances[0][ranked_results.index(identity_name)]
                    # Distance is now correctly negated, so smaller is better.
                    print(f"  --> Match FOUND! at rank {match_rank}, Distance: {match_distance:.4f}")
                else:
                    print("  --> Match NOT FOUND in top results.")
                print("-------------------------------------------\n")
                debug_probe_count += 1
            # --- END DEBUGGING ---

            metrics = calculate_metrics(ranked_results, identity_name, list(range(1, k_max + 1)))
            
            all_mrr.append(metrics['mrr'])
            for k in range(1, k_max + 1):
                all_top_k_accuracy[f'top_{k}_accuracy'].append(metrics[f'top_{k}_accuracy'])

        # Aggregate results
        avg_mrr = np.mean(all_mrr) if all_mrr else 0.0
        aggregated_metrics = {'avg_mrr': avg_mrr}
        for k in range(1, k_max + 1):
            key = f'avg_top_{k}_accuracy'
            current_k_accuracies = all_top_k_accuracy[f'top_{k}_accuracy']
            aggregated_metrics[key] = np.mean(current_k_accuracies) if current_k_accuracies else 0.0

        print(f"DEBUG: Aggregated metrics for {exp_description}: {aggregated_metrics}") # Keep this debug print
        return aggregated_metrics

    except Exception as e: # <--- ADD THIS EXCEPTION BLOCK
        print(f"\nERROR in run_evaluation_experiment for {exp_description}: {e}")
        import traceback
        traceback.print_exc()
        return None # Explicitly return None on error to confirm this path


# --- Main Execution ---

if __name__ == "__main__":
    # Ensure figures directory exists
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)

    # Ensure storage directory exists
    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR)

    # Ensure the tar file is placed in ROOT_DIR
    if not os.path.exists(DATA_ARCHIVE):
        print(f"Error: Dataset archive not found at {DATA_ARCHIVE}. Please place multi_image_identities.tar there.")
        exit()

    # Extract and organize data
    gallery_data, probe_data = extract_and_organize_data(DATA_ARCHIVE, STORAGE_DIR)

    # Initialize augmented preprocessor
    aug_preprocessor = AugmentationPreprocessor()

    # Define common k-values for accuracy evaluation (up to 20 for Task 5)
    k_values_for_accuracy = list(range(1, 21)) # Top-1 to Top-20 for comprehensive analysis

    # Open the markdown file for writing (will overwrite if exists)
    with open(OUTPUT_MD_FILE, 'w') as md_file:
        append_to_markdown(md_file, "# Case Study: Visual Search System Evaluation\n")
        append_to_markdown(md_file, "This document presents a comprehensive evaluation of the IronClad Visual Search System, analyzing different embedding models and the impact of image augmentations on performance. All experiments utilize the brute-force indexing strategy and are evaluated using Mean Reciprocal Rank (MRR) and Top-N Accuracy.\n")
        
        # --- Task 1: Compare Models (Baseline) ---
        append_to_markdown(md_file, "\n## Task 1: Model Comparison (Baseline Performance)\n")
        append_to_markdown(md_file, "This task compares the baseline performance of 'casia-webface' and 'vggface2' pretrained models across different similarity measures.\n")
        
        task1_results_data = []
        best_model_overall = {'mrr': -1, 'top_1': -1, 'name': '', 'metric': ''}

        for model_type, pretrained_name in MODELS.items():
            model_embedding = Embedding(pretrained=pretrained_name, device=DEVICE)
            for metric in SEARCH_METRICS:
                metrics = run_evaluation_experiment(
                    model_type, metric, gallery_data, probe_data, model_embedding, aug_preprocessor,
                    k_max=max(k_values_for_accuracy)
                )
                row_data = {
                    'Model': model_type,
                    'Metric': metric,
                    'Avg MRR': metrics['avg_mrr'],
                    'Avg Top-1 Accuracy': metrics['avg_top_1_accuracy']
                }
                task1_results_data.append(row_data)

                # Determine best model/metric for subsequent tasks (based on MRR, then Top-1)
                if metrics['avg_mrr'] > best_model_overall['mrr']:
                    best_model_overall['mrr'] = metrics['avg_mrr']
                    best_model_overall['top_1'] = metrics['avg_top_1_accuracy']
                    best_model_overall['name'] = model_type
                    best_model_overall['metric'] = metric
                elif math.isclose(metrics['avg_mrr'], best_model_overall['mrr'], rel_tol=1e-9) and \
                     metrics['avg_top_1_accuracy'] > best_model_overall['top_1']:
                    best_model_overall['mrr'] = metrics['avg_mrr']
                    best_model_overall['top_1'] = metrics['avg_top_1_accuracy']
                    best_model_overall['name'] = model_type
                    best_model_overall['metric'] = metric

        append_to_markdown(md_file, create_markdown_table_str(
            ['Model', 'Metric', 'Avg MRR', 'Avg Top-1 Accuracy'], 
            task1_results_data, 
            caption="Table 1: Baseline Performance Comparison of Models and Search Metrics."
        ))
        
        append_to_markdown(md_file, "\n")
        append_to_markdown(md_file, "Based on the baseline results, the best performing model is **{}** with the **{}** metric.\n".format(best_model_overall['name'], best_model_overall['metric']))
        append_to_markdown(md_file, "\n")
        append_to_markdown(md_file, "\n")

        # Set best model and metric for subsequent tasks based on Task 1 results
        BEST_MODEL_NAME = best_model_overall['name']
        BEST_METRIC = best_model_overall['metric']
        best_model_embedding = Embedding(pretrained=MODELS[BEST_MODEL_NAME], device=DEVICE)

        # --- Task 2: Gaussian Noise Impact ---
        append_to_markdown(md_file, "\n## Task 2: Impact of Gaussian Noise Adjustments\n")
        append_to_markdown(md_file, f"This task measures the robustness of the selected best model ({BEST_MODEL_NAME} with {BEST_METRIC}) to various levels of Gaussian noise.\n")
        
        task2_results_data = []
        noise_stds = [0.0, 0.05, 0.1, 0.2] # Varying std deviations for noise
        for std in noise_stds:
            augmentation_params = {'std': std}
            metrics = run_evaluation_experiment(
                BEST_MODEL_NAME, BEST_METRIC, gallery_data, probe_data, best_model_embedding, aug_preprocessor,
                augmentation_type='gaussian_noise', augmentation_params=augmentation_params,
                k_max=max(k_values_for_accuracy)
            )
            row_data = {
                'Noise Std Dev': std,
                'Avg MRR': metrics['avg_mrr'],
                'Avg Top-1 Accuracy': metrics['avg_top_1_accuracy']
            }
            task2_results_data.append(row_data)

        append_to_markdown(md_file, create_markdown_table_str(
            ['Noise Std Dev', 'Avg MRR', 'Avg Top-1 Accuracy'], 
            task2_results_data, 
            caption="Table 2: Performance of {} with {} Metric under Gaussian Noise.".format(BEST_MODEL_NAME, BEST_METRIC)
        ))
        append_to_markdown(md_file, "\n")
        append_to_markdown(md_file, "\n")
        append_to_markdown(md_file, "\n")


        # --- Task 3: Resizing Impact ---
        append_to_markdown(md_file, "\n## Task 3: Impact of Resizing Adjustments\n")
        append_to_markdown(md_file, f"This task evaluates the robustness of the selected best model ({BEST_MODEL_NAME} with {BEST_METRIC}) to various resizing factors.\n")
        
        task3_results_data = []
        resize_factors = [1.0, 0.75, 0.5, 1.25, 1.5] # Varying resize factors
        for factor in resize_factors:
            augmentation_params = {'size_factor': factor}
            metrics = run_evaluation_experiment(
                BEST_MODEL_NAME, BEST_METRIC, gallery_data, probe_data, best_model_embedding, aug_preprocessor,
                augmentation_type='resize', augmentation_params=augmentation_params,
                k_max=max(k_values_for_accuracy)
            )
            row_data = {
                'Resize Factor': factor,
                'Avg MRR': metrics['avg_mrr'],
                'Avg Top-1 Accuracy': metrics['avg_top_1_accuracy']
            }
            task3_results_data.append(row_data)

        append_to_markdown(md_file, create_markdown_table_str(
            ['Resize Factor', 'Avg MRR', 'Avg Top-1 Accuracy'], 
            task3_results_data, 
            caption="Table 3: Performance of {} with {} Metric under Resizing.".format(BEST_MODEL_NAME, BEST_METRIC)
        ))
        append_to_markdown(md_file, "\n")
        append_to_markdown(md_file, "\n")
        append_to_markdown(md_file, "\n")


        # --- Task 4: Brightness Impact ---
        append_to_markdown(md_file, "\n## Task 4: Impact of Brightness Adjustments\n")
        append_to_markdown(md_file, f"This task assesses the robustness of the selected best model ({BEST_MODEL_NAME} with {BEST_METRIC}) to various brightness adjustments.\n")
        
        task4_results_data = []
        brightness_factors = [1.0, 0.5, 0.25, 1.5, 2.0] # Varying brightness factors
        for factor in brightness_factors:
            augmentation_params = {'factor': factor}
            metrics = run_evaluation_experiment(
                BEST_MODEL_NAME, BEST_METRIC, gallery_data, probe_data, best_model_embedding, aug_preprocessor,
                augmentation_type='brightness', augmentation_params=augmentation_params,
                k_max=max(k_values_for_accuracy)
            )
            row_data = {
                'Brightness Factor': factor,
                'Avg MRR': metrics['avg_mrr'],
                'Avg Top-1 Accuracy': metrics['avg_top_1_accuracy']
            }
            task4_results_data.append(row_data)

        append_to_markdown(md_file, create_markdown_table_str(
            ['Brightness Factor', 'Avg MRR', 'Avg Top-1 Accuracy'], 
            task4_results_data, 
            caption="Table 4: Performance of {} with {} Metric under Brightness Changes.".format(BEST_MODEL_NAME, BEST_METRIC)
        ))
        append_to_markdown(md_file, "\n")
        append_to_markdown(md_file, "\n")
        append_to_markdown(md_file, "\n")


        # --- Task 5: Best N Analysis ---
        append_to_markdown(md_file, "\n## Task 5: Determining the Best N for Candidate Identities\n")
        append_to_markdown(md_file, f"This task analyzes the system's accuracy at returning the correct identity within the top N candidates, using the selected best model ({BEST_MODEL_NAME} with {BEST_METRIC}).\n")
        
        # We already ran this in task 1 with k_max=20, so we can use its results directly if preferred,
        # or run a dedicated baseline experiment here if needed.
        # For simplicity, let's just re-run the baseline for a wider range of K values if not done implicitly.
        # The 'metrics' dict from run_evaluation_experiment will contain avg_top_k_accuracy for all k up to k_max.
        
        # The results for task 5 are in results_task5
        task5_results = run_evaluation_experiment(
            BEST_MODEL_NAME, BEST_METRIC, gallery_data, probe_data, best_model_embedding, aug_preprocessor,
            k_max=max(k_values_for_accuracy) # Ensures we get accuracy up to Top-20
        )
        
        task5_results_data = []
        for k in k_values_for_accuracy: # Iterate through 1 to 20
            task5_results_data.append({
                'N': k,
                'Avg Top-N Accuracy': task5_results[f'avg_top_{k}_accuracy']
            })

        append_to_markdown(md_file, create_markdown_table_str(
            ['N', 'Avg Top-N Accuracy'], 
            task5_results_data, 
            caption="Table 5: Average Top-N Accuracy for {} with {} Metric.".format(BEST_MODEL_NAME, BEST_METRIC)
        ))
        append_to_markdown(md_file, "\n")
        append_to_markdown(md_file, "\n")
        append_to_markdown(md_file, "\n")
        
        append_to_markdown(md_file, "\n\nEvaluation complete. Open `CaseStudy.md` in your project folder to fill in your analysis and add figures.")

    print(f"\nEvaluation complete. Results written to {OUTPUT_MD_FILE}")
    print(f"Remember to fill in your interpretations and generate figures in '{FIGURES_DIR}'.")