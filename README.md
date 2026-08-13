# DeepCluster brain-model analysis

Code and lightweight derived data used to analyse the Algonauts 92-image fMRI dataset and generate the publication figures.

## Data policy

This repository does not include the large network-activation files or large pickle result files. Network activations should be obtained separately through QUB PURE. The fMRI data are available through the Algonauts project. The small image-name metadata file is included in `data/`.

The RDM-generation script converts externally supplied activations into lightweight NumPy `.npy` RDM files. These generated RDMs are also not required to be committed; they can be regenerated locally.

## Configuration

The scripts use paths relative to the repository by default. Set these environment variables when the external data live elsewhere:

```bash
export ALGONAUTS_DATA_ROOT=/path/to/algonauts
export ACTIVATIONS_ROOT=/path/to/network/activations_niko92_imgs
export RDM_ROOT=/path/to/generated/rdms
```

`ALGONAUTS_DATA_ROOT` must contain:

```text
Training_Data/92_Image_Set/target_fmri.mat
Training_Data/92_Image_Set/92images/jpg_images/
```

The default Python environment is the `multiple_dc_analysis` conda environment used during development. Install the scientific Python dependencies required by the scripts, including NumPy, pandas, SciPy, Matplotlib, Seaborn, scikit-learn, statsmodels, scikit-bio, pingouin, h5py, hdf5storage, and Pillow.

## Reproduction order

1. Place the activation files supplied through QUB PURE under `ACTIVATIONS_ROOT`.
2. Generate network RDMs:

   ```bash
   python code/get_network_rdms.py
   ```

   This writes `.npy` RDMs under `RDM_ROOT` and diagnostic RDM images under `figures/rdm_plots/`.

3. Compute layer noise ceilings:

   ```bash
   python code/layers_noise_ceiling.py
   ```

4. Compute brain correlations for participant and instance variability:

   ```bash
   python code/corr_with_brain.py
   ```

5. Run the bootstrap regression analysis and its statistics:

   ```bash
   python code/bootstrapping_models.py
   python code/bootstrapping_stats.py
   ```

   The regression bootstrap output is stored as compressed NumPy data rather than pickle.

6. Run the separate mediation bootstrap and significance analysis:

   ```bash
   python code/mediation_separate_models.py
   python code/mediation_separate_models_plots.py
   python code/mediation_ttests.py
   ```

7. Generate Figures 1-6:

   ```bash
   python code/paper_figures.py
   ```

The computational bootstrap scripts are expensive. The repository may optionally provide their CSV/NPZ outputs as a separate archive so that readers can reproduce the figures without rerunning the full bootstrap.

## Main scripts

- `code/get_network_rdms.py`: converts network activations into image-by-image representational dissimilarity matrices.
- `code/layers_noise_ceiling.py`: computes layer-specific noise ceilings across DeepCluster instances.
- `code/corr_with_brain.py`: computes participant- and instance-variability correlations with EVC and IT RDMs.
- `code/bootstrapping_models.py`: performs the NNLS bootstrap regression analysis.
- `code/bootstrapping_stats.py`: computes bootstrap confidence intervals and paired comparisons.
- `code/mediation_separate_models.py`: performs the separate mediation bootstrap.
- `code/mediation_separate_models_plots.py`: creates mediation panels and corrected significance annotations.
- `code/paper_figures.py`: generates the final publication figures.
