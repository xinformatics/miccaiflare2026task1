# Run from the repository root: python3 scripts/check_code_only.py
import pathlib
import subprocess
suffixes = (
    '.pt', '.pth', '.ckpt', '.safetensors', '.onnx', '.h5', '.hdf5',
    '.pkl', '.pickle', '.joblib', '.bin', '.engine', '.tflite', '.pb',
    '.nii', '.nii.gz', '.nrrd', '.mha', '.mhd', '.dcm', '.npy', '.npz',
    '.zip', '.tar', '.tar.gz', '.tgz', '.7z', '.rar',
)
files = subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\0')
bad = [name for name in files if name and (
    name.lower().endswith(suffixes) or pathlib.Path(name).stat().st_size > 2_000_000
)]
if bad:
    raise SystemExit('Move these artifacts to external storage/Zenodo:\n' + '\n'.join(bad))
print('All tracked files satisfy the code-only policy.')
