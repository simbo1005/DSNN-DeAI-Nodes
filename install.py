"""Install the pinned De-AI custom-node stack with the user's exact overrides."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
PROTECTED = ('torch', 'torchvision', 'torchaudio', 'torchcodec', 'triton', 'sageattention', 'numpy')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*args, cwd=None, capture=True, timeout=1800):
    result = subprocess.run(list(map(str,args)), cwd=cwd, text=True,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.STDOUT, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f'{Path(str(args[0])).name} failed ({result.returncode}):\n'+(result.stdout or '')[-4000:])
    return (result.stdout or '').strip()


def safe_relative(root, name):
    part = PurePosixPath(name)
    if not name or part.is_absolute() or any(p in ('..','.') for p in part.parts) or '\\' in name:
        raise RuntimeError(f'Unsafe relative path: {name!r}')
    path = Path(root).joinpath(*part.parts)
    for candidate in (path,*path.parents):
        if candidate == Path(root):break
        if candidate.is_symlink():raise RuntimeError(f'Preserving symlink: {candidate}')
    if not path.resolve().is_relative_to(Path(root).resolve()):raise RuntimeError('Path escapes its package')
    return path


def load_lock():
    lock = json.loads((ROOT/'nodes.lock.json').read_text())
    if lock.get('schema_version') != 1 or len(lock.get('packages',[])) != 4:
        raise RuntimeError('Unsupported node manifest')
    names=set()
    for package in lock['packages']:
        name=package['name']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',name) or name in names:
            raise RuntimeError('Invalid/duplicate package name')
        names.add(name)
        if not re.fullmatch(r'https://github\.com/[^/\s]+/[^/\s]+\.git',package['repository']):
            raise RuntimeError('Packages must use a public GitHub HTTPS origin')
        if not re.fullmatch(r'[a-f0-9]{40}',package['commit']):raise RuntimeError('A package ref is not pinned')
        for overlay in package['overrides']:
            source=safe_relative(ROOT,overlay['source'])
            safe_relative(ROOT,overlay['destination'])
            if digest(source)!=overlay['sha256']:raise RuntimeError('The bundled override changed: '+overlay['source'])
            if not re.fullmatch(r'[a-f0-9]{64}',overlay['upstream_sha256']):raise RuntimeError('Invalid upstream hash')
        performance=package.get('performance_source')
        if performance:
            safe_relative(ROOT,performance['path'])
            if not re.fullmatch(r'[a-f0-9]{64}',performance['sha256']):raise RuntimeError('Invalid performance source hash')
    return lock


def normal_origin(value):
    return value.strip().rstrip('/').removesuffix('.git').lower()


def find_existing(custom_nodes, package):
    matches=[]
    for folder in custom_nodes.iterdir():
        if not folder.is_dir() or not (folder/'.git').exists():continue
        try:origin=run('git','-C',folder,'remote','get-url','origin',timeout=30)
        except RuntimeError:continue
        if normal_origin(origin)==normal_origin(package['repository']):
            if folder.is_symlink():raise RuntimeError('Preserving a linked node checkout: '+str(folder))
            matches.append(folder)
    if len(matches)>1:raise RuntimeError('Duplicate checkouts found for '+package['name']+'; remove the duplicate before installing.')
    if matches:return matches[0]
    preferred=custom_nodes/package['name']
    if preferred.exists() or preferred.is_symlink():
        raise RuntimeError('Preserving an unrecognized existing folder: '+str(preferred))
    return None


def validate_checkout(folder, package):
    if run('git','-C',folder,'rev-parse','HEAD',timeout=30)!=package['commit']:
        raise RuntimeError(f'{folder} is on another revision. It was preserved. Use a fresh ComfyUI/custom_nodes folder or explicitly manage that checkout before retrying.')
    allowed={overlay['destination'] for overlay in package['overrides']}
    changed=set(run('git','-C',folder,'diff','HEAD','--name-only',timeout=30).splitlines())
    if changed-allowed:raise RuntimeError('Preserving local edits in '+str(folder))
    if not (folder/'__init__.py').is_file():raise RuntimeError('Incomplete node package: '+str(folder))
    for overlay in package['overrides']:
        path=safe_relative(folder,overlay['destination'])
        if not path.is_file() or digest(path) not in {overlay['sha256'],overlay['upstream_sha256']}:
            raise RuntimeError('Preserving an unexpected local patch: '+str(path))


def apply_overrides(folder, package, backup_root):
    # Preflight all replacement files before changing any of them.
    validate_checkout(folder,package)
    for overlay in package['overrides']:
        path=safe_relative(folder,overlay['destination'])
        if digest(path)==overlay['sha256']:continue
        backup=Path(backup_root)/package['name']/overlay['upstream_sha256']/overlay['destination']
        backup.parent.mkdir(parents=True,exist_ok=True)
        if backup.exists() and digest(backup)!=overlay['upstream_sha256']:
            raise RuntimeError('An existing backup changed; preserving it: '+str(backup))
        if not backup.exists():shutil.copy2(path,backup)
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.dsnn-',delete=False) as output:
            temporary=Path(output.name)
        try:
            shutil.copy2(safe_relative(ROOT,overlay['source']),temporary)
            os.replace(temporary,path)
        finally:temporary.unlink(missing_ok=True)


def comfy_root(explicit=None):
    selected=explicit or os.environ.get('COMFYUI_DIR')
    candidates=[Path(selected)] if selected else [ROOT.parent.parent,Path('/workspace/ComfyUI'),Path('/workspace/runpod-slim/ComfyUI'),Path('/opt/ComfyUI')]
    for path in candidates:
        path=path.expanduser().resolve()
        if (path/'main.py').is_file() and (path/'folder_paths.py').is_file():return path
    raise RuntimeError('ComfyUI was not found. Pass --comfyui /path/to/ComfyUI.')


def choose_python(comfy, explicit=None):
    if explicit:return str(Path(explicit).expanduser().absolute())
    configured=os.environ.get('COMFYUI_VENV')
    if configured:
        executable=Path(configured)/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        if not executable.is_file():raise RuntimeError('COMFYUI_VENV has no Python executable; pass --python explicitly.')
        return str(executable)
    for name in ('.venv-cu128','.venv','venv'):
        candidate=comfy/name/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        if candidate.is_file():return str(candidate)
    if Path('/venv/main/bin/python').is_file():return '/venv/main/bin/python'
    return sys.executable


def installed_versions(python):
    source="import importlib.metadata as m,json; names="+repr(PROTECTED)+"; found={};\nfor n in names:\n try: found[n]=m.version(n)\n except m.PackageNotFoundError: pass\nprint(json.dumps(found))"
    return json.loads(run(python,'-c',source,timeout=60))


def install_dependencies(python, folders):
    versions=installed_versions(python)
    if 'torch' not in versions:raise RuntimeError('This Python has no PyTorch. Use ComfyUI\'s Python via --python; the installer will not choose a CUDA build for you.')
    with tempfile.TemporaryDirectory(prefix='dsnn-deai-deps-') as directory:
        directory=Path(directory);constraints=directory/'constraints.txt';requirements=directory/'requirements.txt'
        constraints.write_text(''.join(f'{name}=={version}\n' for name,version in versions.items()))
        lines=[]
        for folder in folders:
            source=folder/'requirements.txt'
            if source.is_file():
                for line in source.read_text().splitlines():
                    value=line.strip()
                    if value and not value.startswith('#'):
                        if value.startswith('-') or '://' in value or ' @ ' in value:
                            raise RuntimeError('A pinned upstream requirement needs review: '+value)
                        lines.append(value)
        requirements.write_text('\n'.join(dict.fromkeys(lines))+'\n')
        # CLI constraint supplements image constraints and preserves installed
        # Torch/CUDA/attention ABI versions, including direct wheel installs.
        run(python,'-m','pip','install','--disable-pip-version-check','--only-binary=:all:','--constraint',constraints,'-r',requirements,capture=False)
    current=installed_versions(python)
    if any(current.get(name)!=version for name,version in versions.items()):
        raise RuntimeError('Protected Python versions changed; stop and inspect the environment.')


def ensure_exiftool():
    if not shutil.which('exiftool'):
        if sys.platform.startswith('linux') and shutil.which('apt-get') and os.geteuid()==0:
            run('apt-get','-o','Acquire::Retries=3','update',capture=False)
            run('apt-get','install','-y','--no-install-recommends','libimage-exiftool-perl',capture=False)
        else:
            raise RuntimeError('ExifTool is required. On Debian/Ubuntu install libimage-exiftool-perl (with sudo if needed); on Windows/macOS install ExifTool and add it to PATH, then retry.')
    run('exiftool','-ver',timeout=30)


@contextmanager
def install_lock(comfy):
    state=comfy/'.dsnn-deai';state.mkdir(exist_ok=True)
    path=state/'install.lock'
    try:descriptor=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:raise RuntimeError('Another installation is running. If it was interrupted, remove '+str(path)+' after confirming its process has stopped.')
    try:
        with os.fdopen(descriptor,'w') as handle:handle.write(str(os.getpid()))
        yield state
    finally:path.unlink(missing_ok=True)


def install_runtime(custom, state, receipt, lock, enabled):
    """Install an owned helper package; the four node sources remain untouched."""
    destination=custom/'DSNN-DeAI-Performance'
    if destination.is_symlink():raise RuntimeError('Preserving a symlinked performance helper')
    history=state/'runtime-files.json'
    previous=json.loads(history.read_text()) if history.is_file() else {}
    config={'enabled':enabled,'targets':{}}
    for package,installed in zip(lock['packages'],receipt):
        spec=package.get('performance_source')
        if not spec:continue
        target=safe_relative(Path(installed['path']),spec['path'])
        if digest(target)!=spec['sha256']:raise RuntimeError('Performance source check failed: '+str(target))
        for node in package['required_node_types']:
            config['targets'][node]={'path':str(target.resolve()),'sha256':spec['sha256']}
    contents={name:(ROOT/'runtime'/name).read_bytes() for name in ('__init__.py','performance.py')}
    contents['LICENSE.LC123-MIT.txt']=(ROOT/'LICENSES/LC123-MIT.txt').read_bytes()
    contents['LICENSE.DSNN-MIT.txt']=(ROOT/'LICENSE').read_bytes()
    contents['targets.json']=(json.dumps(config,indent=2)+'\n').encode()
    if destination.exists():
        unknown={p.name for p in destination.iterdir()}-set(contents)-{'__pycache__'}
        if unknown:raise RuntimeError('Preserving an unrecognized performance helper folder')
        for name,content in contents.items():
            path=destination/name
            if path.is_symlink():raise RuntimeError('Preserving a symlinked performance file')
            if path.exists() and digest(path) not in {previous.get(name),hashlib.sha256(content).hexdigest()}:
                raise RuntimeError('Preserving a modified performance helper: '+str(path))
    destination.mkdir(exist_ok=True)
    for name,content in contents.items():
        path=destination/name
        with tempfile.NamedTemporaryFile(dir=destination,prefix='.dsnn-',delete=False) as output:
            output.write(content);temporary=Path(output.name)
        try:os.replace(temporary,path)
        finally:temporary.unlink(missing_ok=True)
    history.write_text(json.dumps({name:digest(destination/name) for name in contents},indent=2)+'\n')


def install(comfy, python, lock, skip_deps=False, skip_system=False, performance=True):
    custom=comfy/'custom_nodes';custom.mkdir(exist_ok=True)
    with install_lock(comfy) as state, tempfile.TemporaryDirectory(prefix='staging-',dir=state) as temporary:
        plans=[]
        for package in lock['packages']:
            existing=find_existing(custom,package)
            folder=existing or Path(temporary)/package['name']
            if not existing:
                print('Fetching',package['name'],package['commit'],flush=True)
                run('git','clone','--filter=blob:none','--no-checkout',package['repository'],folder)
                run('git','-C',folder,'fetch','--depth=1','origin',package['commit'])
                run('git','-C',folder,'checkout','--detach',package['commit'])
            validate_checkout(folder,package)
            plans.append((package,folder,existing))
        if not skip_system:ensure_exiftool()
        if not skip_deps:install_dependencies(python,[folder for _,folder,_ in plans])
        receipt=[]
        for package,folder,existing in plans:
            apply_overrides(folder,package,state/'backups')
            destination=existing or custom/package['name']
            if not existing:folder.rename(destination)
            receipt.append({'name':package['name'],'commit':package['commit'],'path':str(destination),'overrides':{p['destination']:p['sha256'] for p in package['overrides']}})
        install_runtime(custom,state,receipt,lock,performance)
        (state/'installed.json').write_text(json.dumps({'packages':receipt,'python':python,'dependencies_checked':not skip_deps,'exiftool_checked':not skip_system},indent=2)+'\n')
    print('All four packages installed. Restart ComfyUI, then refresh the browser.',flush=True)


def verify(url, lock):
    if urlsplit(url).scheme not in ('http','https'):raise RuntimeError('Use an HTTP(S) ComfyUI URL')
    with urlopen(url.rstrip('/')+'/object_info',timeout=30) as response:registry=json.load(response)
    required=[*lock['core_node_types'],*(node for package in lock['packages'] for node in package['required_node_types'])]
    missing=[name for name in required if name not in registry]
    if missing:raise RuntimeError('ComfyUI did not register: '+', '.join(missing)+'. Restart it and inspect its startup log.')
    schema=registry['RadianceCompressionArtifacts']['input']['optional']['block_size'][1]
    if schema.get('min')!=1 or schema.get('step')!=1:raise RuntimeError('ComfyUI is still using the original Radiance camera.py; restart or remove duplicate packages.')
    print('Verified all 10 workflow node types and patched block_size=1 schema.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfyui');parser.add_argument('--python')
    parser.add_argument('--verify',metavar='COMFYUI_URL',help='Only check node registration after a restart')
    parser.add_argument('--skip-deps',action='store_true',help='Advanced: skip Python package installation/checks')
    parser.add_argument('--skip-system-tools',action='store_true',help='Advanced: skip ExifTool installation/check')
    parser.add_argument('--no-performance',action='store_true',help='Install the original node behavior without the optional runtime optimizations')
    options=parser.parse_args();lock=load_lock()
    if options.verify:verify(options.verify,lock);return
    comfy=comfy_root(options.comfyui);python=choose_python(comfy,options.python)
    print('ComfyUI:',comfy,'\nPython:',python,flush=True)
    install(comfy,python,lock,options.skip_deps,options.skip_system_tools,not options.no_performance)


if __name__=='__main__':
    try:main()
    except (RuntimeError,OSError,subprocess.TimeoutExpired) as error:
        print('Installation stopped:',error,file=sys.stderr);sys.exit(1)
