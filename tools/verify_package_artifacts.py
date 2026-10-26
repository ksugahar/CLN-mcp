"""Verify built CLN wheel/sdist contents and retain artifact identity (stdlib only)."""
import argparse
import ast
from email.parser import BytesParser
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import tarfile
import zipfile

RESOURCES = ('cln/_version.py', 'cln_mcp_server/resources/DiffusionCLN.wl',
             'cln_mcp_server/resources/RoundWireMaxwell.wl')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify(root, directory):
    """Require both formats, matching Version and byte-exact packaged resources."""
    source = root / 'src/cln/_version.py'
    assignments = [ast.literal_eval(n.value) for n in ast.parse(source.read_text(encoding='utf-8')).body
                   if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '__version__' for t in n.targets)]
    if len(assignments) != 1 or not isinstance(assignments[0], str):
        raise ValueError('one source version required')
    version = assignments[0]
    artifacts = []
    for pattern, kind in [('*.whl', 'wheel'), ('*.tar.gz', 'sdist')]:
        paths = list(directory.glob(pattern))
        if len(paths) != 1:
            raise ValueError('exactly one ' + kind + ' required')
        path = paths[0]
        if kind == 'wheel':
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                metadata = [n for n in names if n.endswith('.dist-info/METADATA')]
                if len(metadata) != 1:
                    raise ValueError('one wheel METADATA required')
                actual = BytesParser().parsebytes(archive.read(metadata[0]))['Version']
                for name in RESOURCES:
                    if names.count(name) != 1 or archive.read(name) != (root / 'src' / name).read_bytes():
                        raise ValueError('wheel missing/changed resource: ' + name)
        else:
            with tarfile.open(path) as archive:
                names = archive.getnames()
                metadata = [n for n in names if n.count('/') == 1 and n.endswith('/PKG-INFO')]
                if len(metadata) != 1:
                    raise ValueError('one sdist PKG-INFO required')
                actual = BytesParser().parsebytes(archive.extractfile(metadata[0]).read())['Version']
                for name in RESOURCES:
                    matches = [n for n in names if n.endswith('/src/' + name)]
                    if len(matches) != 1 or archive.extractfile(matches[0]).read() != (root / 'src' / name).read_bytes():
                        raise ValueError('sdist missing/changed resource: ' + name)
        if actual != version:
            raise ValueError(kind + ' Version differs: ' + str((actual, version)))
        artifacts.append({'name': path.name, 'kind': kind, 'bytes': path.stat().st_size,
                          'sha256': digest(path), 'version': actual})
    return {'schema': 'cln.package-artifacts.v1', 'version': version, 'artifacts': artifacts,
            'resources': [{'path': name, 'source_sha256': digest(root / 'src' / name)} for name in RESOURCES]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = verify(root, args.artifact_dir)
    result['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    result['python'] = platform.python_version()
    result['hatchling'] = importlib.metadata.version('hatchling')
    output = args.artifact_dir / 'artifact-manifest.json'
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
