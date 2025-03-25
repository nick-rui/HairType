import os
import cv2
import numpy as np
from tqdm import tqdm
from segment_anything import SamPredictor, sam_model_registry
from skimage.transform import resize
from options import Options

def pad_and_resize(img, size=512):
    '''
    Pad and resize image to square dimensions.
    '''

    img = np.array(img)
    
    if img.shape[0] > img.shape[1]:
        pad_width = int((img.shape[0] - img.shape[1]) / 2)
        img_pad = np.zeros((img.shape[0], pad_width, img.shape[2]))
        padded_img = np.concatenate((img_pad, img, img_pad), axis=1)
    else:
        pad_height = int((img.shape[1] - img.shape[0]) / 2)
        img_pad = np.zeros((pad_height, img.shape[1], img.shape[2]))
        padded_img = np.concatenate((img_pad, img, img_pad), axis=0)
    
    padded_img = resize(padded_img, (size, size)).astype(np.uint8)
    return cv2.merge([padded_img[:,:,0], padded_img[:,:,1], padded_img[:,:,2]])


def write_mask(mask, filename):
    '''
    Write mask to file.
    '''

    if mask.shape[0] == 3:
        mask = mask.transpose(1, 2, 0)
        mask = ((mask[:,:,0] + mask[:,:,1] + mask[:,:,2]) > 0)[:,:,None]
        mask = np.concatenate([mask, mask, mask], axis=2)
    else:
        mask = mask.transpose(1, 2, 0) > 0
    cv2.imwrite(filename, mask * 255)


def generate_masks(image_path, output_path, predictor):
    '''
    Generate hair and body masks using SAM.
    '''

    image = cv2.imread(image_path)
    if image is None:
        return None, None
    
    predictor.set_image(image)
    
    # Generate body mask
    body_masks, _, _ = predictor.predict(
        point_coords=np.array([[255, 255]]),
        point_labels=np.array([1]),
        multimask_output=True
    )
    
    # Generate hair mask
    hair_pt_x = np.where(np.sum(body_masks[:,:,255], axis=0) > 0)[0][0]
    pts = np.array([[255, hair_pt_x + 5], [255, 255]])
    hair_masks, _, _ = predictor.predict(
        point_coords=pts,
        point_labels=np.array([1, 0]),
        multimask_output=False
    )
    
    return hair_masks, body_masks


def apply_mask(image, mask):
    '''
    Apply mask to image with white background.
    '''

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    masked_image = cv2.bitwise_and(image, mask_3channel)
    white_background = np.ones_like(image) * 255
    inverse_mask = cv2.bitwise_not(mask_3channel)
    background = cv2.bitwise_and(white_background, inverse_mask)
    return cv2.add(masked_image, background)


def get_hair_patch(image, mask, patch_size, min_coverage=0.3, max_coverage=0.7):
    '''
    Extract a hair patch from the masked image with coverage between min_coverage and max_coverage.
    Uses a randomized approach to get patches and guarantees an output with at least min_coverage.
    
    Args:
        image: The source image
        mask: Binary mask showing hair regions
        patch_size: Size of the square patch to extract
        min_coverage: Minimum fraction of the patch that should be hair (default: 0.5)
        max_coverage: Maximum fraction of the patch that should be hair (default: 1.0)
    
    Returns:
        A patch of the specified size or None if no suitable patch was found
    '''
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    y_coords, x_coords = np.where(mask > 0)
    
    if len(y_coords) < patch_size * patch_size * min_coverage:
        return None
    
    # Get potential starting points (any point where mask > 0)
    valid_points = list(zip(y_coords, x_coords))
    
    # Shuffle the points to introduce randomness
    import random
    random.shuffle(valid_points)
    
    half_size = patch_size // 2
    total_pixels = patch_size * patch_size
    min_hair_pixels = int(min_coverage * total_pixels)
    max_hair_pixels = int(max_coverage * total_pixels)
    
    # First try using random hair points as centers
    for center_y, center_x in valid_points:
        start_y = center_y - half_size
        start_x = center_x - half_size
        
        # Check if patch would be within image bounds
        if (start_y < 0 or start_x < 0 or 
            start_y + patch_size > mask.shape[0] or 
            start_x + patch_size > mask.shape[1]):
            continue
        
        # Extract the patch from the mask
        mask_patch = mask[start_y:start_y + patch_size, 
                        start_x:start_x + patch_size]
        
        # Count hair pixels in the patch
        hair_pixels = np.sum(mask_patch > 0)
        
        # Check if coverage is within the desired range
        if min_hair_pixels <= hair_pixels <= max_hair_pixels:
            return image[start_y:start_y + patch_size, 
                       start_x:start_x + patch_size]
    
    # If we couldn't find a patch with the exact coverage range, try to find one with at least min_coverage
    for center_y, center_x in valid_points:
        start_y = center_y - half_size
        start_x = center_x - half_size
        
        if (start_y < 0 or start_x < 0 or 
            start_y + patch_size > mask.shape[0] or 
            start_x + patch_size > mask.shape[1]):
            continue
        
        mask_patch = mask[start_y:start_y + patch_size, 
                        start_x:start_x + patch_size]
        
        hair_pixels = np.sum(mask_patch > 0)
        
        if hair_pixels >= min_hair_pixels:
            return image[start_y:start_y + patch_size, 
                       start_x:start_x + patch_size]
    
    # As a fallback, use the center of mass approach from the original function
    center_y = int(np.mean(y_coords))
    center_x = int(np.mean(x_coords))
    
    # Try different offsets around the center of mass
    offsets = [(y, x) for y in range(-half_size, half_size) 
                      for x in range(-half_size, half_size)]
    random.shuffle(offsets)  # Randomize the order
    
    for offset_y, offset_x in offsets:
        start_y = center_y + offset_y - half_size
        start_x = center_x + offset_x - half_size
        
        if (start_y < 0 or start_x < 0 or 
            start_y + patch_size > mask.shape[0] or 
            start_x + patch_size > mask.shape[1]):
            continue
        
        mask_patch = mask[start_y:start_y + patch_size, 
                        start_x:start_x + patch_size]
        
        hair_pixels = np.sum(mask_patch > 0)
        
        if hair_pixels >= min_hair_pixels:
            return image[start_y:start_y + patch_size, 
                       start_x:start_x + patch_size]
    
    return None


def enhance_contrast(image, clip_limit, tile_grid_size):
    '''
    Enhance contrast using CLAHE and contrast stretching.
    '''

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, 
        tileGridSize=(tile_grid_size, tile_grid_size)
    )
    cl = clahe.apply(l)
    
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    lab = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    l_stretched = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
    
    enhanced_lab = cv2.merge((l_stretched, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def process_images(opt):
    '''
    Main processing pipeline.
    '''

    # Initialize SAM if needed
    if not opt.skip_mask_generation:
        print("Initializing SAM model...")
        sam = sam_model_registry[opt.model_type_sam](checkpoint=opt.checkpoint_sam)
        sam.to(device="cpu")
        predictor = SamPredictor(sam)
    
    images = [f for f in os.listdir(opt.input_dir) 
             if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Processing {len(images)} images...")
    
    for img_name in tqdm(images):
        input_path = os.path.join(opt.input_dir, img_name)
        
        # Resize images
        image = cv2.imread(input_path)
        if image is None:
            print(f"Could not load image: {input_path}")
            continue
        
        resized = pad_and_resize(image, opt.image_size)
        cv2.imwrite(os.path.join(opt.resized_dir, img_name), resized)
        
        # Generate masks
        if not opt.skip_mask_generation:
            hair_masks, body_masks = generate_masks(
                os.path.join(opt.resized_dir, img_name),
                opt.output_dir,
                predictor
            )
            if hair_masks is not None:
                write_mask(hair_masks, os.path.join(opt.mask_dir, img_name))
            if body_masks is not None:
                write_mask(body_masks, os.path.join(opt.body_mask_dir, img_name))
        
        # Apply masks
        hair_mask = cv2.imread(os.path.join(opt.mask_dir, img_name), 
                             cv2.IMREAD_GRAYSCALE)
        if hair_mask is not None:
            masked = apply_mask(resized, hair_mask)
            cv2.imwrite(os.path.join(opt.masked_dir, img_name), masked)
        
        # Extract patches
        if not opt.skip_patch_extraction:
            patch = get_hair_patch(resized, hair_mask, opt.patch_size)
            if patch is not None:
                cv2.imwrite(os.path.join(opt.patches_dir, img_name), 
                          patch)
        
        # Enhance contrast
        if not opt.skip_contrast_enhancement:
            patch_path = os.path.join(opt.patches_dir, img_name)
            if os.path.exists(patch_path):
                patch = cv2.imread(patch_path)
                if patch is not None:
                    enhanced = enhance_contrast(
                        patch,
                        opt.clahe_clip_limit,
                        opt.clahe_grid_size
                    )
                    cv2.imwrite(
                        os.path.join(opt.enhanced_dir, img_name),
                        enhanced
                    )

if __name__ == "__main__":
    opt = Options().parse()
    process_images(opt)