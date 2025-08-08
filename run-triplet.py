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
device = "cuda:0" if config['training']['gpu'] and torch.cuda.is_available() else "cpu"

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

def trimodal_triplet_loss(image_features, sketch_features, text_features, margin=0.2, lambda_reg=0.01):
    """
    Compute trimodal triplet loss with a regularization term to ensure gradients flow.
    
    Args:
        image_features: Image features from CLIP (batch_size x dim)
        sketch_features: Sketch features from CLIP (batch_size x dim)
        text_features: Text features from CLIP (batch_size x dim)
        margin: Margin for triplet loss
        lambda_reg: Weight for regularization term
        
    Returns:
        Triplet loss with regularization
    """
    batch_size = image_features.shape[0]
    device = image_features.device
    
    # Compute pairwise distances
    # 1 - cosine similarity = distance
    image_sketch_sim = torch.matmul(image_features, sketch_features.t())
    image_text_sim = torch.matmul(image_features, text_features.t())
    sketch_text_sim = torch.matmul(sketch_features, text_features.t())
    
    # Create mask for positives (diagonal) and negatives (off-diagonal)
    pos_mask = torch.eye(batch_size, device=device)
    neg_mask = 1.0 - pos_mask
    
    # Compute positive and negative similarities
    image_sketch_pos = (image_sketch_sim * pos_mask).sum(dim=1)
    image_text_pos = (image_text_sim * pos_mask).sum(dim=1)
    sketch_text_pos = (sketch_text_sim * pos_mask).sum(dim=1)
    
    # Get hardest negative for each anchor
    image_sketch_neg = (image_sketch_sim * neg_mask).max(dim=1)[0]
    image_text_neg = (image_text_sim * neg_mask).max(dim=1)[0]
    sketch_text_neg = (sketch_text_sim * neg_mask).max(dim=1)[0]
    
    # Compute triplet losses with a small epsilon to ensure non-zero gradients
    epsilon = 1e-6
    
    # Image-Sketch triplet loss
    is_loss = torch.relu(margin - image_sketch_pos + image_sketch_neg + epsilon)
    
    # Image-Text triplet loss
    it_loss = torch.relu(margin - image_text_pos + image_text_neg + epsilon)
    
    # Sketch-Text triplet loss
    st_loss = torch.relu(margin - sketch_text_pos + sketch_text_neg + epsilon)
    
    # Compute mean over non-zero elements to focus on hard examples
    is_loss_mean = is_loss.sum() / (is_loss > 0).sum().float().clamp(min=1.0)
    it_loss_mean = it_loss.sum() / (it_loss > 0).sum().float().clamp(min=1.0)
    st_loss_mean = st_loss.sum() / (st_loss > 0).sum().float().clamp(min=1.0)
    
    # Add regularization term to ensure gradients flow
    # This is a small L2 regularization on the feature vectors
    reg_loss = lambda_reg * (
        image_features.pow(2).sum() +
        sketch_features.pow(2).sum() +
        text_features.pow(2).sum()
    ) / (3 * batch_size)
    
    # Total loss
    triplet_loss = is_loss_mean + it_loss_mean + st_loss_mean + reg_loss
    
    return triplet_loss

optimizer = optim.AdamW(model.parameters(), lr=float(config['model']['learning_rate']), weight_decay=float(config['model']['weight_decay']))

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * config['training']['epochs'])

# Training loop
"""
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

print("Training complete!")"""
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
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        sketch_features = sketch_features / sketch_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute triplet loss
        loss = trimodal_triplet_loss(
            image_features, sketch_features, text_features, 
            margin=0.3, lambda_reg=0.001
        )
        
        # Backward pass
        loss.backward()
        
        # Update weights
        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
        
        scheduler.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    print(f"Epoch {epoch+1}/{config['training']['epochs']}, Train Loss: {train_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch+1}.pt"))

print("Training complete!")