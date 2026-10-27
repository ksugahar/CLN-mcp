"""Artifact boundary contracts; synthetic archives, no package build or FE."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import zipfile
import pytest
from cln._version import __version__

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('verify_package_artifacts', ROOT / 'tools/verify_package_artifacts.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def archives(directory, corrupt=None, fault=None):
    for kind in ('wheel', 'sdist'):
        entries = {name: (ROOT / 'src' / name).read_bytes() for name in verifier.RESOURCES}
        if kind == corrupt:
            if fault == 'missing':
                del entries['cln_mcp_server/resources/RoundWireMaxwell.wl']
            elif fault == 'bytes':
                entries['cln_mcp_server/resources/RoundWireMaxwell.wl'] += b'corrupt'
        version = '999.0' if kind == corrupt and fault == 'version' else __version__
        metadata = ('Metadata-Version: 2.1\nName: CLN-mcp\nVersion: ' + version + '\n').encode()
        if kind == 'wheel':
            with zipfile.ZipFile(directory / 'package.whl', 'w') as archive:
                archive.writestr('package.dist-info/METADATA', metadata)
                for name, data in entries.items():
                    archive.writestr(name, data)
        else:
            with tarfile.open(directory / 'package.tar.gz', 'w:gz') as archive:
                for name, data in {'package/PKG-INFO': metadata, **{'package/src/' + n: b for n, b in entries.items()}}.items():
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))


def test_artifact_manifest_matches_actual_bytes(tmp_path):
    """Guards: retained identities match both files. Assumes: synthetic archives."""
    archives(tmp_path)
    result = verifier.verify(ROOT, tmp_path)
    assert result['version'] == __version__, result
    for item in result['artifacts']:
        data = (tmp_path / item['name']).read_bytes()
        assert item['bytes'] == len(data), item
        assert item['sha256'] == hashlib.sha256(data).hexdigest(), item
    assert len(result['resources']) == 4, result


@pytest.mark.parametrize('kind', ['wheel', 'sdist'])
@pytest.mark.parametrize('fault', ['missing', 'bytes', 'version'])
def test_both_artifacts_reject_resource_or_metadata_corruption(tmp_path, kind, fault):
    """Guards: missing/changed new resource or wrong Version fails. Assumes: no build."""
    archives(tmp_path, kind, fault)
    with pytest.raises(ValueError, match='resource|Version'):
        verifier.verify(ROOT, tmp_path)
