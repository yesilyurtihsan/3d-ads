import argparse
from data.mvtec3d import mvtec3d_classes, get_data_loader
import torch
from tqdm import tqdm
import pandas as pd
from fpfh_features import FPFHFeatures

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FPFH test")
    args = parser.parse_args()

    classes = mvtec3d_classes()

    METHOD_NAMES = [
        "FPFH"
        #"RGB + FPFH"
    ]

    for cls in classes:
        print(f"\nRunning on class {cls}\n")

        # Training
        train_loader = get_data_loader("train", class_name=cls, img_size=224)
        # Adding samples to memory bank
        method = FPFHFeatures()
        for sample, _ in tqdm(train_loader, desc=f"Extracting train features for class {cls}"):
            method.add_sample_to_mem_bank(sample)
        
        print(f"\n\nRunning coreset for {method} on class {cls}...")
        method.run_coreset()



        # # Testing and Evaluating
        # image_rocauc = dict()
        # pixel_rocauc = dict()
        # au_pros = dict()
        # test_loader = get_data_loader("test", cls, img_size=224)
        # with torch.no_grad():
        #     for sample, mask, label in tqdm(test_loader, desc=f"Extracting test features for label {cls}"):
        #         method.predict(sample, mask, label)
        
        # method.calculate_metrics()
        # image_rocauc[method]  = round(method.image_rocauc, 3)
        # pixel_rocauc[method] = round(method.pixel_rocauc, 3)
        # au_pros[method] = round(method.au_pro, 3)