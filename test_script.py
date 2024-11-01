import os
from pathlib import Path
from torchvision.utils import save_image

val_dir = "F:/KLA Problem Statement/Denoising_Dataset_train_val"
results_dir = "F:/KLA Problem Statement/Denoising_Dataset_results"

Path(results_dir).mkdir(parents=True, exist_ok=True)

def process_and_save_images(model, val_dir, results_dir, transform):
    model.eval() 
    with torch.no_grad():
        for obj in os.listdir(val_dir):
            obj_dir = os.path.join(val_dir, obj, 'Val', 'Degraded_image')
            if not os.path.isdir(obj_dir):
                continue
            
            for defect_type in os.listdir(obj_dir):
                defect_dir = os.path.join(obj_dir, defect_type)
                save_defect_dir = os.path.join(results_dir, obj, 'Val', defect_type)
                Path(save_defect_dir).mkdir(parents=True, exist_ok=True)
                for img_name in os.listdir(defect_dir):
                    if img_name.endswith('.png'):
                        degraded_img_path = os.path.join(defect_dir, img_name)
                        mask_img_path = os.path.join(
                            val_dir, obj, 'Val', 'Defect_mask', defect_type, img_name.replace('.png', '_mask.png')
                        )

                        degraded_img = Image.open(degraded_img_path).convert('RGB')
                        mask_img = Image.open(mask_img_path).convert('L')
                        degraded_img = transform(degraded_img).to(device)
                        mask_img = transform(mask_img).to(device)
                        mask_img = mask_img.unsqueeze(0)  
                        output = model(degraded_img.unsqueeze(0), mask_img).squeeze(0)
                        output_path = os.path.join(save_defect_dir, img_name)
                        save_image(output, output_path)

transform = transforms.Compose([
    transforms.ToTensor(),
])


process_and_save_images(model_trained, val_dir, results_dir, transform)