"""Configurable paths for the publication analysis pipeline."""

import os
import pickle
from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get('ALGONAUTS_DATA_ROOT', PROJECT_ROOT / 'data'))
ACTIVATIONS_ROOT = Path(os.environ.get('ACTIVATIONS_ROOT', DATA_ROOT / 'activations_niko92_imgs'))
RDM_ROOT = Path(os.environ.get('RDM_ROOT', PROJECT_ROOT / 'data' / 'rdms'))
RESULTS_ROOT = PROJECT_ROOT / 'results'
FIGURES_ROOT = PROJECT_ROOT / 'figures'
IMAGE_NAMES_PATH = PROJECT_ROOT / 'data' / 'niko92_img_names.pickle'
FMRI_PATH = DATA_ROOT / 'Training_Data' / '92_Image_Set' / 'target_fmri.mat'
IMAGE_ROOT = DATA_ROOT / 'Training_Data' / '92_Image_Set' / '92images' / 'jpg_images'

for directory in (RDM_ROOT, RESULTS_ROOT, FIGURES_ROOT):
    directory.mkdir(parents=True, exist_ok=True)


def rdm_path(name):
    """Return the preferred RDM path, accepting legacy pickle files."""
    npy_path = RDM_ROOT / f'{name}.npy'
    pickle_path = RDM_ROOT / f'{name}.pickle'
    return npy_path if npy_path.exists() or not pickle_path.exists() else pickle_path


def load_rdm(name):
    path = rdm_path(name)
    if path.suffix == '.npy':
        return np.load(path)
    with path.open('rb') as handle:
        return pickle.load(handle)


def load_bootstrap_values():
    """Load regression bootstrap values, preferring compressed NumPy data."""
    npz_path = RESULTS_ROOT / 'CombinedRegressionModels_BootstrappingValues_NNLS.npz'
    pickle_path = RESULTS_ROOT / 'CombinedRegressionModels_BootstrappingValues_NNLS.pickle'
    path = npz_path if npz_path.exists() else pickle_path
    if path.suffix == '.npz':
        archive = np.load(path)
        return {key: archive[key] for key in archive.files}
    with path.open('rb') as handle:
        return pickle.load(handle, encoding='latin1')
