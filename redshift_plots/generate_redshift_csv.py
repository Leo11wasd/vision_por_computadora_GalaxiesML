
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from galaxy_loader import GalaxyDataset


def main():
    # Define las rutas para cargar el dataset y el modelo
    dataset_path='/Users/isaacapm/Documents/notas/especialidad/tecnicas_de_vision_por_computadora/laboratorios/proyecto/dataset/DESI_HSC_64x64_v2.hdf5'
    model_uri = "/Users/isaacapm/Documents/notas/especialidad/tecnicas_de_vision_por_computadora/laboratorios/proyecto/modelos/model_artifacts_v2/best_model"

    # ——————————————————————————————————————————————
    # 1) Instantiate & load your new model
    # ——————————————————————————————————————————————
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    # Step 2: Load model using MLflow
    print(device)
    mmae = mlflow.pytorch.load_model(model_uri, map_location=device)
    print(mmae)

    """Print model summary"""
    total_params = sum(p.numel() for p in mmae.parameters())
    trainable_params = sum(p.numel() for p in mmae.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ——————————————————————————————————————————————
    # 2) Dataset / DataLoader
    # ——————————————————————————————————————————————
    dataset = GalaxyDataset(
        h5_path=dataset_path,
        transform = None, normalize=True)

    total_size = len(dataset)

    # Define split sizes
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size  # Remaining 15%

    g = torch.Generator().manual_seed(0)
    _, _, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=g)

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    y_test = np.zeros(test_size)
    y_pred = np.zeros(test_size)
    mmae.eval()
    mmae.to(device)
    dataset = test_loader.dataset # 0 - imagen, 1 - spectro, 2 - redshift
    with torch.no_grad():
        for index, sample in enumerate(dataset):
            if isinstance(sample, (list, tuple)):
                if len(sample) == 2:
                    img, spec = sample
                    redshift, targetid = None, None
                elif len(sample) == 3:
                    img, spec, redshift = sample
                    targetid = None
                elif len(sample) == 4:
                    img, spec, redshift, targetid = sample
                else:
                    raise ValueError(f"Unexpected dataset sample length {len(sample)}")
            else:
                raise TypeError(f"Unexpected sample type: {type(sample)}")
            
            # Add batch dimension
            img = img.unsqueeze(0)
            spec = spec.unsqueeze(0)

            img, spec = img.to(device), spec.to(device)

            # reshape spectrum
            spec = spec.unsqueeze(1)  # (B,1,7783)
            spec = F.avg_pool1d(spec, kernel_size=30, stride=30)
            outputs = mmae(img, spec)
            y_test[index] = redshift.numpy()
            y_pred[index] = outputs['redshift_pred'].cpu().numpy()[0]

    redshift_df = pd.DataFrame()
    redshift_df['y_test'] = y_test
    redshift_df['y_pred'] = y_pred
    redshift_df.to_csv('mmae_redshift_predictions_v2.csv', index=False)


if __name__ == '__main__':
    main()