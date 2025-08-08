import torch
from torch import nn, optim
import os
from tqdm import tqdm
from src import clip
from dataloader import get_dataloader
import yaml
from utils import load_config, convert_models_to_fp32

# Load configuration
config = load_config('config.yml')

# Set up device
device = "cuda:1" if config['training']['gpu'] and torch.cuda.is_available() else "cpu"

# Load CLIP model
model, preprocess = clip.load(config['model']['model_name'], device=device, jit=False)

# Check for existing checkpoints
save_path = config['training']['save_model_path']
checkpoints = [f for f in os.listdir(save_path) if f.startswith("model_epoch_") and f.endswith(".pt")]
start_epoch = 0

if checkpoints:
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    checkpoint_path = os.path.join(save_path, latest_checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    start_epoch = int(latest_checkpoint.split("_")[-1].split(".")[0])
    print(f"Resuming from epoch {start_epoch}")

# Prepare data loaders     
train_loader = get_dataloader(
    config['data']['csv_path'],
    config['data']['image_path'],
    config['data']['sketch_path'],
    config['model']['batch_size'],
    'train',
    preprocess,
    num_workers=config['data']['num_workers']
)

'''val_loader = get_dataloader(
    config['data']['csv_path'],
    config['data']['image_path'],
    config['data']['sketch_path'],
    config['model']['batch_size'],
    'val',
    preprocess,
    num_workers=config['data']['num_workers'],
    shuffle=False
)'''

# Loss and optimizer
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

def compute_trimodal_triplet_loss(image_features, sketch_features, text_features, margin=0.2):
    """
    Computes a trimodal triplet loss between image, sketch, and text modalities.
    
    Args:
        image_features: Normalized image features (B x D)
        sketch_features: Normalized sketch features (B x D)
        text_features: Normalized text features (B x D)
        margin: Margin value for triplet loss (default: 0.2)
        
    Returns:
        Triplet loss averaged across all modality combinations
    """
    batch_size = image_features.shape[0]
    device = image_features.device
    
    # Initialize loss
    triplet_loss = 0.0
    
    # Image as anchor, with sketch as positive and all other sketches as negatives
    image_sketch_pos = torch.sum(image_features * sketch_features, dim=1)
    image_sketch_dist = 1.0 - image_sketch_pos.unsqueeze(1)  # Distance = 1 - cosine similarity
    
    # Create a mask for valid negatives (all except the positive)
    mask = torch.ones((batch_size, batch_size), device=device) - torch.eye(batch_size, device=device)
    
    # For each anchor, find its hardest negative
    image_sketch_neg_dist, _ = (mask.unsqueeze(2) * image_sketch_dist.unsqueeze(1)).min(dim=1)
    
    # Compute image-sketch triplet loss with margin
    image_sketch_loss = torch.relu(image_sketch_dist + margin - image_sketch_neg_dist).mean()
    triplet_loss += image_sketch_loss
    
    # Image as anchor, with text as positive and all other texts as negatives
    image_text_pos = torch.sum(image_features * text_features, dim=1)
    image_text_dist = 1.0 - image_text_pos.unsqueeze(1)
    
    # For each anchor, find its hardest negative
    image_text_neg_dist, _ = (mask.unsqueeze(2) * image_text_dist.unsqueeze(1)).min(dim=1)
    
    # Compute image-text triplet loss with margin
    image_text_loss = torch.relu(image_text_dist + margin - image_text_neg_dist).mean()
    triplet_loss += image_text_loss
    
    # Sketch as anchor, with image as positive and all other images as negatives
    sketch_image_loss = torch.relu(image_sketch_dist + margin - image_sketch_neg_dist).mean()
    triplet_loss += sketch_image_loss
    
    # Sketch as anchor, with text as positive and all other texts as negatives
    sketch_text_pos = torch.sum(sketch_features * text_features, dim=1)
    sketch_text_dist = 1.0 - sketch_text_pos.unsqueeze(1)
    
    # For each anchor, find its hardest negative
    sketch_text_neg_dist, _ = (mask.unsqueeze(2) * sketch_text_dist.unsqueeze(1)).min(dim=1)
    
    # Compute sketch-text triplet loss with margin
    sketch_text_loss = torch.relu(sketch_text_dist + margin - sketch_text_neg_dist).mean()
    triplet_loss += sketch_text_loss
    
    # Text as anchor, with image as positive and all other images as negatives
    text_image_loss = torch.relu(image_text_dist + margin - image_text_neg_dist).mean()
    triplet_loss += text_image_loss
    
    # Text as anchor, with sketch as positive and all other sketches as negatives
    text_sketch_loss = torch.relu(sketch_text_dist + margin - sketch_text_neg_dist).mean()
    triplet_loss += text_sketch_loss
    
    # Average the loss across all six combinations
    triplet_loss = triplet_loss / 6.0
    
    return triplet_loss


optimizer = optim.AdamW(model.parameters(), lr=float(config['model']['learning_rate']), weight_decay=float(config['model']['weight_decay']))

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * config['training']['epochs'])

# Training loop

for epoch in range(start_epoch, config['training']['epochs']):
    model.train()
    train_loss = 0.0
    
    for images, sketches, texts in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}"):
        images = images.to(device)
        sketches = sketches.to(device)
        texts = texts.to(device)
        
        optimizer.zero_grad()
        
        image_features = model.encode_image(images)
        sketch_features = model.encode_image(sketches)
        text_features = model.encode_text(texts)
        
        # Normalize features
        #image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        #sketch_features = sketch_features / sketch_features.norm(dim=-1, keepdim=True)
        #text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Scaled pairwise cosine similarities
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ sketch_features.t()
        logits_per_text = logit_scale * image_features @ text_features.t()
        
        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
        
        total_loss = (
            loss_img(logits_per_image, ground_truth) +
            loss_img(logits_per_image.t(), ground_truth) +
            loss_txt(logits_per_text, ground_truth) +
            loss_txt(logits_per_text.t(), ground_truth)
        ) / 4
        
        total_loss.backward()
        
        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
        
        scheduler.step()
        
        train_loss += total_loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    '''
    with torch.no_grad():
        for images, sketches, texts in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            sketches = sketches.to(device)
            texts = texts.to(device)
            
            image_features = model.encode_image(images)
            sketch_features = model.encode_image(sketches)
            text_features = model.encode_text(texts)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            sketch_features = sketch_features / sketch_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Scaled pairwise cosine similarities
            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ sketch_features.t()
            logits_per_text = logit_scale * image_features @ text_features.t()
            
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            
            total_loss = (
                loss_img(logits_per_image, ground_truth) +
                loss_img(logits_per_image.t(), ground_truth) +
                loss_txt(logits_per_text, ground_truth) +
                loss_txt(logits_per_text.t(), ground_truth)
            ) / 4
            
            val_loss += total_loss.item()
    
    val_loss /= len(val_loader)'''
    
    print(f"Epoch {epoch+1}/{config['training']['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch+1}.pt"))

print("Training complete!")
"""
# Training loop
for epoch in range(start_epoch, config['training']['epochs']):
    model.train()
    train_loss = 0.0
    contrastive_loss_total = 0.0
    triplet_loss_total = 0.0
    
    for images, sketches, texts in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}"):
        images = images.to(device)
        sketches = sketches.to(device)
        texts = texts.to(device)
        
        optimizer.zero_grad()
        
        image_features = model.encode_image(images)
        sketch_features = model.encode_image(sketches)
        text_features = model.encode_text(texts)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        sketch_features = sketch_features / sketch_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Scaled pairwise cosine similarities
        logit_scale = model.logit_scale.exp()
        logits_per_image_sketch = logit_scale * image_features @ sketch_features.t()
        logits_per_image_text = logit_scale * image_features @ text_features.t()
        
        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
        
        # Original contrastive loss
        contrastive_loss = (
            loss_img(logits_per_image_sketch, ground_truth) +
            loss_img(logits_per_image_sketch.t(), ground_truth) +
            loss_txt(logits_per_image_text, ground_truth) +
            loss_txt(logits_per_image_text.t(), ground_truth)
        ) / 4
        
        # Trimodal triplet loss
        triplet_loss = compute_trimodal_triplet_loss(
            image_features, sketch_features, text_features, margin=0.2
        )
        
        # Combine both losses
        total_loss = contrastive_loss + triplet_loss
        
        total_loss.backward()
        
        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
        
        scheduler.step()
        
        train_loss += total_loss.item()
        contrastive_loss_total += contrastive_loss.item()
        triplet_loss_total += triplet_loss.item()
    
    train_loss /= len(train_loader)
    contrastive_loss_total /= len(train_loader)
    triplet_loss_total /= len(train_loader)
    
    print(f"Epoch {epoch+1}/{config['training']['epochs']}, Train Loss: {train_loss:.4f}, "
          f"Contrastive Loss: {contrastive_loss_total:.4f}, Triplet Loss: {triplet_loss_total:.4f}")
    
    # Save model
    torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch+1}.pt"))

print("Training complete!")"""