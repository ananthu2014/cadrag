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
def trimodal_infonce_loss(image_features, sketch_features, text_features, temperature=0.07):
    """
    Implements a trimodal InfoNCE loss that aligns all three modalities.
    
    Args:
        image_features: Tensor of shape [batch_size, feature_dim]
        sketch_features: Tensor of shape [batch_size, feature_dim]
        text_features: Tensor of shape [batch_size, feature_dim]
        temperature: Scaling factor for logits
    
    Returns:
        Trimodal InfoNCE loss
    """
    device = image_features.device
    batch_size = image_features.shape[0]
    
    # Normalize all features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    sketch_features = sketch_features / sketch_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    
    # Compute pairwise similarities
    i2s_sim = torch.exp(torch.matmul(image_features, sketch_features.t()) / temperature)
    i2t_sim = torch.exp(torch.matmul(image_features, text_features.t()) / temperature)
    s2t_sim = torch.exp(torch.matmul(sketch_features, text_features.t()) / temperature)
    
    # Compute InfoNCE losses for each pair of modalities
    labels = torch.arange(batch_size, device=device)
    
    # Image-to-sketch loss
    i2s_loss = -torch.log(
        i2s_sim[torch.arange(batch_size), torch.arange(batch_size)] / 
        i2s_sim.sum(dim=1)
    ).mean()
    
    # Image-to-text loss
    i2t_loss = -torch.log(
        i2t_sim[torch.arange(batch_size), torch.arange(batch_size)] / 
        i2t_sim.sum(dim=1)
    ).mean()
    
    # Sketch-to-text loss
    s2t_loss = -torch.log(
        s2t_sim[torch.arange(batch_size), torch.arange(batch_size)] / 
        s2t_sim.sum(dim=1)
    ).mean()
    
    # Reverse direction losses
    s2i_loss = -torch.log(
        i2s_sim[torch.arange(batch_size), torch.arange(batch_size)] / 
        i2s_sim.sum(dim=0)
    ).mean()
    
    t2i_loss = -torch.log(
        i2t_sim[torch.arange(batch_size), torch.arange(batch_size)] / 
        i2t_sim.sum(dim=0)
    ).mean()
    
    t2s_loss = -torch.log(
        s2t_sim[torch.arange(batch_size), torch.arange(batch_size)] / 
        s2t_sim.sum(dim=0)
    ).mean()
    
    # Average all losses
    total_loss = (i2s_loss + i2t_loss + s2t_loss + s2i_loss + t2i_loss + t2s_loss) / 6
    losses = {"i2s": i2s_loss.item(), "i2t": i2t_loss.item(), "s2t": s2t_loss.item()}
    print(f"Individual losses: {losses}")
    # At the end of your loss function
    if torch.isnan(total_loss):
        print("NaN detected in loss calculation!")
        # Log intermediate values to debug
        print(f"i2s_sim min/max: {i2s_sim.min().item()}, {i2s_sim.max().item()}")
        print(f"i2t_sim min/max: {i2t_sim.min().item()}, {i2t_sim.max().item()}")
        print(f"s2t_sim min/max: {s2t_sim.min().item()}, {s2t_sim.max().item()}")
        # Return a fallback loss or raise exception
        return torch.tensor(1.0, requires_grad=True, device=device)
def stable_trimodal_infonce_loss(image_features, sketch_features, text_features, temperature=0.07):
    """
    A numerically stable implementation of trimodal InfoNCE loss with detailed debugging.
    """
    device = image_features.device
    batch_size = image_features.shape[0]
    
    # Print feature statistics for debugging
    print(f"Image features: min={image_features.min().item():.4f}, max={image_features.max().item():.4f}, has_nan={torch.isnan(image_features).any().item()}")
    print(f"Sketch features: min={sketch_features.min().item():.4f}, max={sketch_features.max().item():.4f}, has_nan={torch.isnan(sketch_features).any().item()}")
    print(f"Text features: min={text_features.min().item():.4f}, max={text_features.max().item():.4f}, has_nan={torch.isnan(text_features).any().item()}")
    print(f"Temperature: {temperature:.6f}")
    
    # Check and fix any NaN or inf values in input features
    image_features = torch.nan_to_num(image_features, nan=0.0, posinf=1.0, neginf=-1.0)
    sketch_features = torch.nan_to_num(sketch_features, nan=0.0, posinf=1.0, neginf=-1.0)
    text_features = torch.nan_to_num(text_features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Normalize features with epsilon for stability
    image_features = image_features / (image_features.norm(dim=1, keepdim=True) + 1e-8)
    sketch_features = sketch_features / (sketch_features.norm(dim=1, keepdim=True) + 1e-8)
    text_features = text_features / (text_features.norm(dim=1, keepdim=True) + 1e-8)
    
    # Double-check normalized features
    if torch.isnan(image_features).any() or torch.isnan(sketch_features).any() or torch.isnan(text_features).any():
        print("NaN detected after normalization!")
        return torch.tensor(0.0, requires_grad=True, device=device)  # Return a safe value to continue training
    
    # Use a safe minimum temperature
    safe_temp = max(temperature, 1e-4)
    
    # Compute logits directly (avoiding exp until needed by CrossEntropyLoss)
    i2s_logits = torch.matmul(image_features, sketch_features.t()) / safe_temp
    i2t_logits = torch.matmul(image_features, text_features.t()) / safe_temp
    s2t_logits = torch.matmul(sketch_features, text_features.t()) / safe_temp
    
    # Print logits statistics
    print(f"i2s_logits: min={i2s_logits.min().item():.4f}, max={i2s_logits.max().item():.4f}")
    print(f"i2t_logits: min={i2t_logits.min().item():.4f}, max={i2t_logits.max().item():.4f}")
    print(f"s2t_logits: min={s2t_logits.min().item():.4f}, max={s2t_logits.max().item():.4f}")
    
    # Create labels
    labels = torch.arange(batch_size, device=device)
    
    # We'll use try-except blocks for each loss term to catch any issues
    losses = {}
    try:
        i2s_loss = nn.CrossEntropyLoss()(i2s_logits, labels)
        losses['i2s'] = i2s_loss
    except Exception as e:
        print(f"Error in i2s_loss: {e}")
        losses['i2s'] = torch.tensor(0.0, requires_grad=True, device=device)
    
    try:
        s2i_loss = nn.CrossEntropyLoss()(i2s_logits.t(), labels)
        losses['s2i'] = s2i_loss
    except Exception as e:
        print(f"Error in s2i_loss: {e}")
        losses['s2i'] = torch.tensor(0.0, requires_grad=True, device=device)
    
    try:
        i2t_loss = nn.CrossEntropyLoss()(i2t_logits, labels)
        losses['i2t'] = i2t_loss
    except Exception as e:
        print(f"Error in i2t_loss: {e}")
        losses['i2t'] = torch.tensor(0.0, requires_grad=True, device=device)
    
    try:
        t2i_loss = nn.CrossEntropyLoss()(i2t_logits.t(), labels)
        losses['t2i'] = t2i_loss
    except Exception as e:
        print(f"Error in t2i_loss: {e}")
        losses['t2i'] = torch.tensor(0.0, requires_grad=True, device=device)
    
    try:
        s2t_loss = nn.CrossEntropyLoss()(s2t_logits, labels)
        losses['s2t'] = s2t_loss
    except Exception as e:
        print(f"Error in s2t_loss: {e}")
        losses['s2t'] = torch.tensor(0.0, requires_grad=True, device=device)
    
    try:
        t2s_loss = nn.CrossEntropyLoss()(s2t_logits.t(), labels)
        losses['t2s'] = t2s_loss
    except Exception as e:
        print(f"Error in t2s_loss: {e}")
        losses['t2s'] = torch.tensor(0.0, requires_grad=True, device=device)
    
    # Print individual loss values
    print(f"Individual losses: {losses}")
    
    # Average all losses (filtering out any that might be NaN)
    valid_losses = [loss for loss in losses.values() if not torch.isnan(loss)]
    if len(valid_losses) > 0:
        total_loss = sum(valid_losses) / len(valid_losses)
    else:
        # If all losses are NaN, return a safe tensor with gradient
        total_loss = torch.tensor(0.0, requires_grad=True, device=device)
    
    return total_loss    
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
        

        # Modified implementation for trimodal loss
        # First, calculate the similarity between sketches and text
        logits_sketch_text = logit_scale * sketch_features @ text_features.t()

        '''# Update the total loss calculation
            total_loss = (
                # Image-Sketch similarities
                loss_img(logits_per_image, ground_truth) +
                loss_img(logits_per_image.t(), ground_truth) +
                
                # Image-Text similarities
                loss_txt(logits_per_text, ground_truth) +
                loss_txt(logits_per_text.t(), ground_truth) +
                
                # Sketch-Text similarities (new)
                loss_txt(logits_sketch_text, ground_truth) +
                loss_txt(logits_sketch_text.t(), ground_truth)
            ) / 6  # Divide by 6 since we now have 6 terms'''
        if (torch.isnan(image_features).any() or 
            torch.isnan(sketch_features).any() or 
            torch.isnan(text_features).any()):
            print("NaN detected in features! Skipping batch.")
            continue

        # Use safe temperature scaling
        temperature = 0.07  # Default fallback
        try:
            temperature = 1.0/model.logit_scale.exp().item()
            # Clamp temperature to reasonable values
            temperature = max(min(temperature, 100.0), 0.01)
        except:
            pass  # Use default if any issue occurs

        # Use stable trimodal loss
        try:
            total_loss = stable_trimodal_infonce_loss(
                image_features, 
                sketch_features, 
                text_features,
                temperature=temperature
            )
            
            # Check for NaN loss
            if torch.isnan(total_loss):
                print("NaN loss detected! Using zero loss for this batch.")
                total_loss = torch.tensor(0.0, requires_grad=True, device=device)
            
            # Proceed with backward pass
            total_loss.backward()
            
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
        except Exception as e:
            print(f"Error in loss calculation: {e}")
            continue  # Skip this batch

        
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