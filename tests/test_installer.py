import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('deai_install',ROOT/'install.py')
installer=importlib.util.module_from_spec(spec);spec.loader.exec_module(installer)


def checkout(tmp_path,package):
    folder=tmp_path/'package';folder.mkdir()
    subprocess.run(['git','init','-q',str(folder)],check=True)
    (folder/'__init__.py').write_text('NODE_CLASS_MAPPINGS={}\n')
    for item in package['overrides']:
        path=folder/item['destination'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'original source\n')
        item['upstream_sha256']=installer.digest(path)
    subprocess.run(['git','-C',str(folder),'add','.'],check=True)
    subprocess.run(['git','-C',str(folder),'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture'],check=True)
    package['commit']=installer.run('git','-C',folder,'rev-parse','HEAD')
    return folder


def test_manifest_exactly_matches_workflow_packages_and_node_ids():
    lock=installer.load_lock()
    assert [p['required_node_types'][0] for p in lock['packages']]==['Vintage Effect','BondSaveWithCustomMetadata','LCLensProfile','RadianceCompressionArtifacts']
    assert len(lock['core_node_types'])==6
    assert {p['destination'] for package in lock['packages'] for p in package['overrides']}=={'bond_nodes.py','film/camera.py'}
    assert {p.name for p in ROOT.iterdir() if p.suffix.lower() in ('.png','.jpg')}==set()
    for package in lock['packages']:
        for overlay in package['overrides']:ast.parse((ROOT/overlay['source']).read_text())


def test_uploaded_files_remain_byte_identical():
    assert installer.digest(ROOT/'overrides/Bond-Node-Suite/bond_nodes.py')=='fd303df9a05775505af0418607fa8724cfa91a40d9737278a2e6ea8fef43a07d'
    assert installer.digest(ROOT/'overrides/radiance/film/camera.py')=='1215b6d1d9e300a5b6e9f870dfe333853028e481f4bf277492655c0e39b33bd3'


def test_patching_backs_up_original_and_is_idempotent(tmp_path):
    package=copy.deepcopy(installer.load_lock()['packages'][1]);folder=checkout(tmp_path,package)
    backups=tmp_path/'backups';installer.apply_overrides(folder,package,backups)
    target=folder/'bond_nodes.py';assert installer.digest(target)==package['overrides'][0]['sha256']
    backup=next(backups.rglob('bond_nodes.py'));assert backup.read_bytes()==b'original source\n'
    installer.apply_overrides(folder,package,backups)
    assert backup.read_bytes()==b'original source\n'


def test_unrecognized_local_patch_is_not_overwritten(tmp_path):
    package=copy.deepcopy(installer.load_lock()['packages'][1]);folder=checkout(tmp_path,package)
    (folder/'bond_nodes.py').write_text('my other edits')
    with pytest.raises(RuntimeError,match='local patch'):installer.apply_overrides(folder,package,tmp_path/'backups')
    assert (folder/'bond_nodes.py').read_text()=='my other edits'
    assert not (tmp_path/'backups').exists()


def test_other_tracked_edits_and_different_versions_are_preserved(tmp_path):
    package=copy.deepcopy(installer.load_lock()['packages'][1]);folder=checkout(tmp_path,package)
    (folder/'__init__.py').write_text('local edits')
    with pytest.raises(RuntimeError,match='local edits'):installer.validate_checkout(folder,package)
    assert (folder/'__init__.py').read_text()=='local edits'
    package['commit']='0'*40
    with pytest.raises(RuntimeError,match='another revision'):installer.validate_checkout(folder,package)


@pytest.mark.parametrize('path',['../escape','/absolute','film/../../escape','film\\camera.py'])
def test_paths_cannot_escape_the_installation(tmp_path,path):
    with pytest.raises(RuntimeError):installer.safe_relative(tmp_path,path)


def test_override_cannot_write_through_symlink(tmp_path):
    outside=tmp_path/'elsewhere';outside.mkdir();folder=tmp_path/'package';folder.mkdir()
    (folder/'film').symlink_to(outside,target_is_directory=True)
    with pytest.raises(RuntimeError,match='symlink'):installer.safe_relative(folder,'film/camera.py')


def test_duplicate_package_folders_are_reported(tmp_path):
    package=copy.deepcopy(installer.load_lock()['packages'][1])
    for name in ('one','two'):
        folder=tmp_path/name;subprocess.run(['git','init','-q',str(folder)],check=True)
        subprocess.run(['git','-C',str(folder),'remote','add','origin',package['repository']],check=True)
    with pytest.raises(RuntimeError,match='Duplicate'):installer.find_existing(tmp_path,package)


def test_python_requirements_constrain_existing_cuda_and_attention_packages(tmp_path,monkeypatch):
    root=tmp_path/'node';root.mkdir();(root/'requirements.txt').write_text('torch>=2.0\nPillow>=10\n')
    before={'torch':'2.10.0+cu128','triton':'3.6.0','sageattention':'2.2.0','numpy':'2.2.6'}
    monkeypatch.setattr(installer,'installed_versions',lambda _:dict(before))
    seen=[]
    def run(*args,**kwargs):
        args=list(map(str,args));seen.extend(args)
        constraints=Path(args[args.index('--constraint')+1]).read_text()
        assert 'torch==2.10.0+cu128' in constraints and 'sageattention==2.2.0' in constraints
        assert '--only-binary=:all:' in args
        return ''
    monkeypatch.setattr(installer,'run',run)
    installer.install_dependencies(sys.executable,[root])
    assert seen[:4]==[sys.executable,'-m','pip','install']


def test_install_lock_is_released_even_on_failure(tmp_path):
    with pytest.raises(RuntimeError,match='synthetic'):
        with installer.install_lock(tmp_path):
            with pytest.raises(RuntimeError,match='Another installation'):
                with installer.install_lock(tmp_path):pass
            raise RuntimeError('synthetic')
    assert not (tmp_path/'.dsnn-deai/install.lock').exists()


def test_runtime_install_update_disable_and_local_edit_preservation(tmp_path):
    custom=tmp_path/'custom_nodes';custom.mkdir();state=tmp_path/'.dsnn-deai';state.mkdir()
    lock=copy.deepcopy(installer.load_lock());receipt=[]
    for package in lock['packages']:
        folder=custom/package['name'];folder.mkdir()
        source=folder/package['performance_source']['path'];source.parent.mkdir(parents=True,exist_ok=True)
        source.write_text('fixture source')
        package['performance_source']['sha256']=installer.digest(source)
        receipt.append({'path':str(folder)})
    installer.install_runtime(custom,state,receipt,lock,True)
    helper=custom/'DSNN-DeAI-Performance'
    config=json.loads((helper/'targets.json').read_text());assert config['enabled']
    assert len(config['targets'])==4 and (helper/'LICENSE.LC123-MIT.txt').is_file()
    installer.install_runtime(custom,state,receipt,lock,False)
    assert json.loads((helper/'targets.json').read_text())['enabled'] is False
    (helper/'performance.py').write_text('my local customization')
    with pytest.raises(RuntimeError,match='modified performance helper'):
        installer.install_runtime(custom,state,receipt,lock,True)
    assert (helper/'performance.py').read_text()=='my local customization'
