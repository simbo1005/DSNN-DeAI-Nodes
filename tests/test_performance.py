"""Compare acceleration with the independently captured pinned LC123 method."""
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
torch=pytest.importorskip('torch')
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dsnn_perf_test',ROOT/'runtime/performance.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
reference=json.loads((ROOT/'tests/fixtures/lens_reference.json').read_text())
env={'torch':torch,'F':F,'math':math,'_LENS_PROFILES':reference['presets'],
     '_preview':lambda self,out,src:{'result':(out,),'ui':{}}}
exec(reference['method'],env)


def baseline(image,lens,mode,strength):
    # Match LC123's existing RGBA decorator around its independent method.
    result=env['run'](object(),image[...,:3],lens,mode,strength)['result'][0]
    if image.shape[-1]==4:result=torch.cat((result,image[...,3:4]),dim=-1)
    return result


@pytest.fixture(autouse=True)
def reset():
    p._maps.clear();p._geo_cache.clear();p._records.clear()
    yield
    p._maps.clear()


@pytest.mark.parametrize('lens',list(reference['presets']))
@pytest.mark.parametrize('mode',['Add Aberrations','Correct Aberrations'])
def test_float32_lens_matches_actual_pinned_algorithm(lens,mode):
    image=torch.rand((2,37,59,3),generator=torch.Generator().manual_seed(200))
    expected=baseline(image,lens,mode,.73)
    actual,hit=p.lens_image(image,reference['presets'][lens],.73,mode,'cpu')
    torch.testing.assert_close(actual,expected,rtol=0,atol=2e-6)
    assert not hit
    again,hit=p.lens_image(image,reference['presets'][lens],.73,mode,'cpu')
    assert hit;assert torch.equal(again,actual)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_rgba_dtype_and_original_alpha_preserved(dtype):
    image=torch.rand((1,43,31,4),generator=torch.Generator().manual_seed(17)).to(dtype)
    lens=next(iter(reference['presets']))
    out,_=p.lens_image(image,reference['presets'][lens],.5,'Add Aberrations','cpu')
    expected=baseline(image,lens,'Add Aberrations',.5)
    assert out.dtype==dtype and out.shape==image.shape
    assert out.is_contiguous()
    assert torch.equal(out[...,3],image[...,3])
    torch.testing.assert_close(out,expected,rtol=0,atol=2e-6 if dtype==torch.float32 else .002)


def test_zero_strength_is_exact_identity():
    image=torch.randn(1,7,9,4)
    result,hit=p.lens_image(image,next(iter(reference['presets'].values())),0,'Add Aberrations','cpu')
    assert result is image and not hit


def test_cache_respects_byte_budget(monkeypatch):
    monkeypatch.setenv('DSNN_DEAI_GRID_CACHE_MB','1')
    cache=p.MapCache()
    small=lambda:(torch.zeros(100),None)
    assert cache.get('small',small)[1] is False
    assert cache.get('small',small)[1] is True
    cache.get('too-large',lambda:(torch.zeros(300000),None))
    assert cache.bytes<=1024**2 and 'too-large' not in cache.entries
    monkeypatch.setenv('DSNN_DEAI_GRID_CACHE_MB','0')
    cache.get('disabled',small)
    assert cache.bytes==0 and not cache.entries


def test_previews_shrink_without_modifying_result_tensor(monkeypatch):
    monkeypatch.setenv('DSNN_DEAI_PREVIEW_EDGE','32');seen=[]
    target=SimpleNamespace(save_images=lambda image,**kw:seen.append(image) or {'ui':{'images':[]}})
    proxy=p.PreviewProxy(target);image=torch.rand(2,96,128,4);before=image.clone()
    proxy.save_images(image,filename_prefix='test')
    assert seen[0].shape==(2,24,32,4)
    assert torch.equal(image,before)
    monkeypatch.setenv('DSNN_DEAI_PREVIEW_EDGE','0');proxy.save_images(image)
    assert seen[-1] is image


def test_metadata_read_reuse_invalidation_and_command_passthrough():
    calls=[]
    def run(args,**kwargs):
        calls.append(list(args));return SimpleNamespace(stdout='[]',returncode=0)
    proxy=p.MetadataProcessProxy(SimpleNamespace(run=run,PIPE='pipe'))
    read=['exiftool','-json','-a','-G1','image.jpg']
    state={'reads':{},'count':0};token=p._scope.set(state)
    try:
        one=proxy.run(read,capture_output=True,text=True)
        assert proxy.run(read,capture_output=True,text=True) is one
        assert len(calls)==1
        proxy.run(['exiftool','-overwrite_original','image.jpg'])
        proxy.run(read,capture_output=True,text=True);proxy.run(read,capture_output=True,text=True)
        assert len(calls)==3 and state['count']==2
    finally:p._scope.reset(token)
    proxy.run(read);proxy.run(read)
    assert len(calls)==5 and proxy.PIPE=='pipe'


def test_geocoding_success_cached_and_failure_retried():
    calls=[]
    def lookup(city):calls.append(city);return (1.0,2.0) if city=='known' else (None,None)
    cached=p.cached_geocoder(lookup)
    assert cached('known')==(1.0,2.0);assert cached('known')==(1.0,2.0)
    cached('failed');cached('failed')
    assert calls==['known','failed','failed']


def test_vintage_single_image_copy_removal_is_pixel_exact():
    proxy=p.SingleImageTorchProxy(torch);image=torch.rand(11,17,3)
    result=proxy.stack([image])
    assert torch.equal(result,torch.stack([image])) and result.is_contiguous()
    assert result.data_ptr()==image.data_ptr()
    batch=proxy.stack([image,image+1])
    assert torch.equal(batch,torch.stack([image,image+1]))


def test_cuda_out_of_memory_uses_original_cpu_method(monkeypatch):
    class Lens:
        def run(self,image,lens,mode,strength=1.0,look='Custom'):
            return {'result':(image+.125,),'ui':{}}
    monkeypatch.setattr(p,'choose_device',lambda _: (torch.device('cuda'),'test'))
    def oom(*args):raise torch.cuda.OutOfMemoryError('synthetic')
    monkeypatch.setattr(p,'lens_image',oom)
    emptied=[];monkeypatch.setattr(torch.cuda,'empty_cache',lambda:emptied.append(True))
    p.accelerate_lens(Lens,SimpleNamespace(_LENS_PROFILES={'test':{}}))
    image=torch.zeros(1,7,9,3);result=Lens().run(image,'test','Add Aberrations')
    assert torch.equal(result['result'][0],image+.125)
    assert emptied==[True] and p._records[-1]['fallback'] is True


def test_async_timing_preserves_coroutine_and_return_value():
    import asyncio,inspect
    class AsyncNode:
        FUNCTION='execute'
        async def execute(self,value):
            await asyncio.sleep(0);return (value+1,)
    p.time_node(AsyncNode,'async test')
    assert inspect.iscoroutinefunction(AsyncNode.execute)
    assert asyncio.run(AsyncNode().execute(41))==(42,)
    assert p._records[-1]['node']=='async test'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires a real CUDA GPU')
def test_cuda_lens_matches_pinned_cpu_reference():
    image=torch.rand((1,83,107,4),generator=torch.Generator().manual_seed(33))
    for lens in reference['presets']:
        actual,_=p.lens_image(image,reference['presets'][lens],.73,'Add Aberrations','cuda')
        expected=baseline(image,lens,'Add Aberrations',.73)
        torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=.001)
