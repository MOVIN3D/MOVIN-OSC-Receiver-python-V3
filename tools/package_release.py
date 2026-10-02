"""Build the user ZIP from an explicit file list, without bundling development files."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

repo = Path(__file__).resolve().parent.parent
version = (repo / 'VERSION').read_text(encoding='utf-8').strip()
if not re.fullmatch(r'\d+\.\d+\.\d+', version):
    raise ValueError('VERSION must contain a semantic version')
if f'v{version}' not in (repo / 'README.md').read_text(encoding='utf-8').splitlines()[0]:
    raise ValueError('README version does not match VERSION')
if f'## {version} ' not in (repo / 'CHANGELOG.md').read_text(encoding='utf-8'):
    raise ValueError('CHANGELOG entry is missing')
files = ('main.py', 'requirements.txt', 'environment.yml', 'README.md', 'CHANGELOG.md', 'VERSION')
output = repo / 'dist' / f'v{version}'
output.mkdir(parents=True, exist_ok=True)
archive_path = output / f'MOVIN-OSC-Receiver-v{version}.zip'
with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for name in files:
        entry = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
        entry.compress_type = zipfile.ZIP_DEFLATED
        entry.external_attr = 0o644 << 16
        archive.writestr(entry, (repo / name).read_bytes())
with zipfile.ZipFile(archive_path) as archive:
    if archive.testzip() is not None or set(archive.namelist()) != set(files):
        raise ValueError('ZIP validation failed')
    for name in files:
        if archive.read(name) != (repo / name).read_bytes():
            raise ValueError(f'ZIP content differs: {name}')
manifest = {
    'version': version,
    'studio_min': '3.0.0',
    'studio_status_min': '3.3.0',
    'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
    'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo)),
    'package': {
        'file': archive_path.name,
        'sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        'bytes': archive_path.stat().st_size,
        'files': {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in files},
    },
}
manifest_path = output / 'manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
(output / 'SHA256SUMS.txt').write_text(''.join(
    f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n'
    for path in (archive_path, manifest_path)), encoding='utf-8')
print(json.dumps(manifest, indent=2))
