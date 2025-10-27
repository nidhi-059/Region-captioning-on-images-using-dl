import torch
import json
import os
import pandas as pd
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from dataset import FlickrDataset, collate_fn # Assuming your dataset file is named dataset.py
from torch.utils.data import DataLoader
from torchvision import transforms

def format_for_coco_eval(captions_df, generated_captions):
    """
    Formats the ground truth and generated captions into the COCO JSON format.
    
    Args:
        captions_df (pd.DataFrame): DataFrame with 'image' and 'caption' columns for ground truth.
        generated_captions (list of dicts): List of {'image_id': str, 'caption': str}.
        
    Returns:
        tuple: (path_to_ground_truth_json, path_to_results_json)
    """
    # --- Create Ground Truth Annotation File ---
    annotations = []
    images = []
    img_id_counter = 0
    unique_images = {}

    for idx, row in captions_df.iterrows():
        img_name = row['image']
        if img_name not in unique_images:
            unique_images[img_name] = img_id_counter
            images.append({'id': img_id_counter, 'file_name': img_name})
            img_id_counter += 1
        
        img_id = unique_images[img_name]
        annotations.append({
            'image_id': img_id,
            'id': idx,
            'caption': row['caption']
        })

    ground_truth_data = {
        'images': images,
        'annotations': annotations,
        'type': 'captions',
        'info': 'Flickr8k ground truth for COCO eval',
        'licenses': ''
    }
    
    gt_path = 'results/ground_truth_coco.json'
    with open(gt_path, 'w') as f:
        json.dump(ground_truth_data, f)

    # --- Create Results File ---
    results_data = []
    for item in generated_captions:
        img_name = item['image_id']
        if img_name in unique_images:
            results_data.append({
                'image_id': unique_images[img_name],
                'caption': item['caption']
            })
            
    res_path = 'results/generated_captions_coco.json'
    with open(res_path, 'w') as f:
        json.dump(results_data, f)
        
    return gt_path, res_path


def evaluate_model(encoder, decoder, data_loader, vocab, device):
    """
    Evaluate the generative model on a dataset split (e.g., validation or test).
    """
    encoder.eval()
    decoder.eval()
    
    generated_captions = []
    # Use the image names from the dataloader's dataset
    image_names_in_split = data_loader.dataset.image_names
    
    print("Generating captions for evaluation...")
    for img_name in tqdm(image_names_in_split):
        # We only need to generate one caption per unique image
        image_path = os.path.join(data_loader.dataset.root_dir, 'Images', img_name)
        
        # This reuses the inference logic, assuming it's in a separate file or defined
        from inference import generate_caption 
        caption = generate_caption(image_path, encoder, decoder, vocab, device)
        
        generated_captions.append({"image_id": img_name, "caption": caption})

    print("Formatting data for COCO evaluation...")
    # Get the ground truth captions for the current split
    gt_df = data_loader.dataset.df[data_loader.dataset.df['image'].isin(image_names_in_split)]
    
    ann_file, res_file = format_for_coco_eval(gt_df, generated_captions)

    print("Running COCO evaluation...")
    coco = COCO(ann_file)
    coco_res = coco.loadRes(res_file)
    
    coco_eval = COCOEvalCap(coco, coco_res)
    
    # Evaluate on all images in the result file
    coco_eval.params['image_id'] = coco_res.getImgIds()
    
    coco_eval.evaluate()

    print("\n--- Evaluation Metrics ---")
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.4f}")
    print("--------------------------")


if __name__ == '__main__':
    # This is an example of how you would run the evaluation.
    # You would load your trained models and a test/validation dataloader here.
    
    # --- Placeholder for loading models and data ---
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # with open('data/vocab.pkl', 'rb') as f:
    #     vocab = pickle.load(f)
    #
    # gen_encoder_cnn = GenerativeEncoderCNN(embed_size).to(device)
    # decoder_mrnn = DecoderMRNN(embed_size, hidden_size, len(vocab)).to(device)
    #
    # gen_encoder_cnn.load_state_dict(torch.load('saved_models/gen_encoder_cnn.pth'))
    # decoder_mrnn.load_state_dict(torch.load('saved_models/decoder_mrnn.pth'))
    #
    # # Create a dataloader for your test/validation set
    # test_dataset = FlickrDataset(...)
    # test_loader = DataLoader(test_dataset,...)
    #
    # evaluate_model(gen_encoder_cnn, decoder_mrnn, test_loader, vocab, device)
    print("Evaluation script is ready. Integrate it into your main workflow.")