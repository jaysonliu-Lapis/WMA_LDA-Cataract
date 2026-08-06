# Dataset card

## Contents

The package contains exactly the 212 cataract_patient_dataset training patients used in the ResNet-50 cataract PROP experiment, one raw right-eye JPEG per patient, complete source-table fields for the selected patients, and the cached features used by the classifier.

`cataract_metadata.csv` preserves every column used from `data.xlsx`:

| Field | Meaning |
| --- | --- |
| `ID` | cataract_patient_dataset patient identifier |
| `Patient Age`, `Patient Sex` | source demographic fields |
| `Left-Fundus`, `Right-Fundus` | source image names |
| `Left-Diagnostic Keywords`, `Right-Diagnostic Keywords` | source diagnostic text |
| `N`, `D`, `G`, `C`, `A`, `H`, `M`, `O` | source patient-level binary disease flags |
| `Y_cataract` | derived outcome: 0 = cataract only; 1 = cataract plus at least one of D/G/A/H/M/O |
| `right_eye_image` | copied image name in `data/images/right/` |

`cataract_labels.csv` is the smaller model-facing table. Its rows and order match `features_resnet50.csv`, `features_resnet50.npy`, and the image directory.

## Feature provenance

The features were produced from the original right-eye JPEGs with the ImageNet-pretrained `torchvision.models.resnet50` model using `ResNet50_Weights.IMAGENET1K_V2`. The final `fc` layer was replaced by `torch.nn.Identity()`, producing 2,048-dimensional GAP features. Input preprocessing follows `weights.transforms()`.

## Label scope and limitation

cataract_patient_dataset labels in this package are patient-level labels. They may reflect a diagnosis made using both eyes, while this study fixes the model input to the right eye. Consequently, this is a patient-level cataract-comorbidity task, not a verified same-eye multi-label image task.

## Integrity and reproducibility

`SHA256SUMS` lists checksums for the data package. The expected class counts are 146 class-0 and 66 class-1 patients. `scripts/package_used_data.py` validates IDs, right-eye filenames, feature dimension, and class membership while rebuilding the package.

## Permission for this release

The collaborating hospital project supplying this study's data package has authorized public release of the selected images, metadata, and derived features with this repository. This permission statement is recorded in [DATA_PERMISSION.md](../DATA_PERMISSION.md). Please cite both the associated paper and the cataract_patient_dataset provenance when reusing the package.
