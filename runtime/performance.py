"""Accelerate the pinned De-AI nodes without rewriting their source or graph.

Lens equations are adapted from LC123's MIT-licensed lc_image_tools.py;
see LICENSES/LC123-MIT.txt in the source repository. JPEG codecs/settings stay
with the original nodes. GPU float32 interpolation can differ in rounding.
"""
from collections import OrderedDict, deque
from contextvars import ContextVar
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import threading
import time

import torch
import torch.nn.functional as F

_records = deque(maxlen=100)
_record_lock = threading.RLock()
_scope = ContextVar('dsnn_deai_metadata_reads', default=None)
_geo_cache = OrderedDict()
_geo_lock = threading.RLock()
_patched = {}
_warnings = {}


def integer_setting(name, default, low, high):
    try:return max(low,min(high,int(os.getenv(name,str(default)))))
    except ValueError:return default


class MapCache:
    def __init__(self):self.entries=OrderedDict();self.bytes=0;self.lock=threading.RLock()
    def clear(self):
        with self.lock:self.entries.clear();self.bytes=0
    def get(self,key,create):
        budget=integer_setting('DSNN_DEAI_GRID_CACHE_MB',512,0,2048)*1024**2
        with self.lock:
            while self.bytes>budget and self.entries:
                _,(_,size)=self.entries.popitem(last=False);self.bytes-=size
            if key in self.entries:
                value,size=self.entries.pop(key);self.entries[key]=(value,size);return value,True
        value=create()
        size=sum(t.numel()*t.element_size() for t in value if t is not None)
        if size<=budget:
            with self.lock:
                # Another invocation can have filled the same key meanwhile.
                previous=self.entries.pop(key,None)
                if previous:self.bytes-=previous[1]
                while self.bytes+size>budget and self.entries:
                    _,(_,old)=self.entries.popitem(last=False);self.bytes-=old
                self.entries[key]=(value,size);self.bytes+=size
        return value,False


_maps=MapCache()


def lens_maps(height,width,preset,strength,mode,device):
    """Same coordinates, reflection padding and align_corners as pinned LC123."""
    p=preset;sign=1.0 if mode=='Add Aberrations' else -1.0
    k1=p['k1']*strength*sign;k2=p['k2']*strength*sign
    ca_r=p['ca_r']*strength*sign;ca_b=p['ca_b']*strength*sign
    vig=p['vig']*strength;vig_mid=p['vig_mid']
    cy,cx=height/2.0,width/2.0;scale=min(height,width)/1024.0
    yy,xx=torch.meshgrid(torch.arange(height,dtype=torch.float32,device=device),
                         torch.arange(width,dtype=torch.float32,device=device),indexing='ij')
    ny=(yy-cy)/max(cy,1e-6);nx=(xx-cx)/max(cx,1e-6)
    r2=nx*nx+ny*ny;r4=r2*r2;r=torch.sqrt(r2+1e-8)
    max_r=math.sqrt(cx*cx+cy*cy)+1e-6;grids=[]
    for ca in (ca_r*scale,0.0,ca_b*scale):
        distort=1.0+k1*r2+k2*r4
        total=distort*(1.0+(ca/max_r)*r) if abs(ca)>0.01 else distort
        src_x=nx*total*cx+cx;src_y=ny*total*cy+cy
        gx=(src_x/max(width-1,1))*2-1;gy=(src_y/max(height-1,1))*2-1
        grids.append(torch.stack((gx,gy),dim=-1))
    vig_mul=None
    if vig>0.01:
        falloff=(1.0/torch.sqrt(1.0+r2))**4
        transition=((r-vig_mid*0.8)/0.4).clamp(0,1)
        if mode=='Add Aberrations':vig_mul=(1.0-transition*(1.0-falloff)*vig*2).clamp(0,1).unsqueeze(-1)
        else:vig_mul=(1.0+transition*(1.0/falloff.clamp(0.3,1.0)-1.0)*vig).unsqueeze(-1)
    return torch.stack(grids),vig_mul


@torch.inference_mode()
def lens_image(image,preset,strength,mode,device):
    if strength<0.01:return image,False
    batch,height,width,channels=image.shape
    if channels not in (3,4):raise ValueError('LC Lens expects an RGB or RGBA image')
    device=torch.device(device)
    key=(height,width,tuple(sorted(preset.items())),float(strength),mode,str(device))
    (grids,vignette),hit=_maps.get(key,lambda:lens_maps(height,width,preset,strength,mode,device))
    rgb=image[...,:3].to(device=device,dtype=torch.float32)
    planes=rgb.permute(0,3,1,2).reshape(batch*3,1,height,width)
    sample_grids=grids.unsqueeze(0).expand(batch,-1,-1,-1,-1).reshape(batch*3,height,width,2)
    out=F.grid_sample(planes,sample_grids,mode='bilinear',padding_mode='reflection',align_corners=True)
    out=out.reshape(batch,3,height,width).permute(0,2,3,1)
    if vignette is not None:out=out*vignette
    out=out.clamp(0,1).to(image.dtype)
    if channels==4:out=torch.cat((out,image[...,3:4].to(device)),dim=-1)
    return out.contiguous(),hit


def choose_device(image):
    mode=os.getenv('DSNN_DEAI_DEVICE','auto').strip().lower()
    if mode=='cpu':return torch.device('cpu'),'CPU requested'
    try:
        from comfy import model_management
        device=model_management.get_torch_device()
        if device.type!='cuda' or not torch.cuda.is_available():return torch.device('cpu'),'ComfyUI is not using CUDA'
        # Include reusable PyTorch allocator blocks without unloading models.
        free,_=torch.cuda.mem_get_info(device)
        free+=max(0,torch.cuda.memory_reserved(device)-torch.cuda.memory_allocated(device))
        required=image.shape[0]*image.shape[1]*image.shape[2]*128+256*1024**2
        if free<required:return torch.device('cpu'),'Insufficient free CUDA memory'
        return device,'CUDA float32'
    except (ImportError,RuntimeError,AssertionError,OSError):return torch.device('cpu'),'CUDA unavailable'


class PreviewProxy:
    def __init__(self,target):self.target=target;self._lc_alpha=None;self.saved=0
    def __getattr__(self,name):return getattr(self.target,name)
    def save_images(self,images,*args,**kwargs):
        edge=integer_setting('DSNN_DEAI_PREVIEW_EDGE',1024,0,8192)
        height,width=images.shape[1:3]
        if edge and max(height,width)>edge:
            ratio=edge/max(height,width)
            images=F.interpolate(images.movedim(-1,1).float(),
                size=(max(1,round(height*ratio)),max(1,round(width*ratio))),mode='area').movedim(1,-1)
        self.saved+=1
        return self.target.save_images(images,*args,**kwargs)


def record(node,started,**details):
    entry={'node':node,'seconds':round(time.perf_counter()-started,6),'timestamp':time.time(),**details}
    try:
        from comfy_execution.utils import get_executing_context
        context=get_executing_context()
        if context is not None:
            entry['node_id']=getattr(context,'node_id',None)
            entry['prompt_id']=getattr(context,'prompt_id',None)
    except Exception:pass # Optional timing context must never block a node.
    with _record_lock:_records.append(entry)
    detail=' '.join(f'{key}={value}' for key,value in details.items())
    print(f"[DSNN De-AI] {node}: {entry['seconds']:.3f}s {detail}",flush=True)


def accelerate_lens(cls,module):
    original=cls.run
    @functools.wraps(original)
    def run(self,image,lens,mode,strength=1.0,look='Custom'):
        started=time.perf_counter();proxy=PreviewProxy(self);fallback=False
        device,reason=(image.device,'Identity strength') if strength<0.01 else choose_device(image)
        try:
            try:
                result,hit=lens_image(image,module._LENS_PROFILES[lens],strength,mode,device)
                output=module._preview(proxy,result,image.to(device))
                output['result']=(output['result'][0].to(image.device),)
            except torch.cuda.OutOfMemoryError:
                if device.type!='cuda':raise
                _maps.clear();torch.cuda.empty_cache();fallback=True;device=torch.device('cpu');reason='CUDA OOM; original CPU fallback'
                output=original(proxy,image,lens,mode,strength,look);hit=False
            record('LCLensProfile',started,device=str(device),reason=reason,grid_cache_hit=hit,
                   preview_edge=integer_setting('DSNN_DEAI_PREVIEW_EDGE',1024,0,8192),fallback=fallback)
            return output
        except BaseException:
            record('LCLensProfile',started,device=str(device),failed=True);raise
    cls.run=run


class MetadataProcessProxy:
    """Cache only Bond's identical EXIF read subprocesses within one save."""
    def __init__(self,original):self.original=original
    def __getattr__(self,name):return getattr(self.original,name)
    def run(self,args,*positional,**kwargs):
        state=_scope.get()
        read=(isinstance(args,(list,tuple)) and len(args)==5 and list(args[1:4])==['-json','-a','-G1'])
        if state is None:return self.original.run(args,*positional,**kwargs)
        if read:
            key=(tuple(map(str,args)),tuple(sorted(kwargs.items())))
            if key not in state['reads']:
                state['reads'][key]=self.original.run(args,*positional,**kwargs);state['count']+=1
            return state['reads'][key]
        result=self.original.run(args,*positional,**kwargs)
        state['reads'].clear() # Metadata may have been written between before/after.
        return result


def cached_geocoder(original):
    @functools.wraps(original)
    def lookup(city):
        now=time.monotonic()
        with _geo_lock:
            previous=_geo_cache.pop(city,None)
            if previous and now-previous[0]<86400:
                _geo_cache[city]=previous;return previous[1]
        value=original(city)
        if value[0] is not None and value[1] is not None:
            with _geo_lock:
                _geo_cache[city]=(now,value)
                while len(_geo_cache)>128:_geo_cache.popitem(last=False)
        return value
    return lookup


def accelerate_bond(cls,module):
    original=getattr(cls,cls.FUNCTION)
    if not isinstance(module.subprocess,MetadataProcessProxy):module.subprocess=MetadataProcessProxy(module.subprocess)
    if not getattr(module._geocode_city,'_dsnn_cached',False):
        module._geocode_city=cached_geocoder(module._geocode_city);module._geocode_city._dsnn_cached=True
    @functools.wraps(original)
    def save(*args,**kwargs):
        started=time.perf_counter();state={'reads':{},'count':0};token=_scope.set(state)
        try:return original(*args,**kwargs)
        finally:
            _scope.reset(token);record('BondSaveWithCustomMetadata',started,metadata_reads=state['count'])
    setattr(cls,cls.FUNCTION,save)


class SingleImageTorchProxy:
    """Avoid Vintage's final full-frame stack copy for a one-image batch."""
    def __init__(self,original):self.original=original
    def __getattr__(self,name):return getattr(self.original,name)
    def stack(self,tensors,dim=0,*,out=None):
        if out is None and len(tensors)==1:return tensors[0].unsqueeze(dim)
        return self.original.stack(tensors,dim=dim,out=out)


def time_node(cls,node):
    original=getattr(cls,cls.FUNCTION)
    if inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        async def execute_async(*args,**kwargs):
            started=time.perf_counter()
            try:return await original(*args,**kwargs)
            finally:record(node,started,backend='upstream')
        setattr(cls,cls.FUNCTION,execute_async)
        return
    @functools.wraps(original)
    def execute(*args,**kwargs):
        started=time.perf_counter()
        try:return original(*args,**kwargs)
        finally:record(node,started,backend='upstream')
    setattr(cls,cls.FUNCTION,execute)


def apply_nodes(registry,config):
    if not config.get('enabled',True) or os.getenv('DSNN_DEAI_PERFORMANCE','1').lower() in ('0','false','off'):return
    for node,spec in config.get('targets',{}).items():
        cls=registry.get(node)
        if cls is None or _patched.get(node)==id(cls):continue
        try:
            path=Path(inspect.getsourcefile(cls)).resolve()
            if path!=Path(spec['path']).resolve() or hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:
                raise ValueError('Source differs from the pinned, reviewed node')
            module=inspect.getmodule(cls)
            if node=='LCLensProfile':accelerate_lens(cls,module)
            elif node=='BondSaveWithCustomMetadata':accelerate_bond(cls,module)
            else:
                if node=='Vintage Effect' and not isinstance(module.torch,SingleImageTorchProxy):
                    module.torch=SingleImageTorchProxy(module.torch)
                time_node(cls,node)
            _patched[node]=id(cls);_warnings.pop(node,None)
        except Exception as error:
            text=str(error)
            if _warnings.get(node)!=text:print(f'[DSNN De-AI] {node}: original node retained ({text})',flush=True)
            _warnings[node]=text
    for node in ('ImageUpscaleWithModel','ImageScaleBy'):
        cls=registry.get(node)
        if cls is not None and _patched.get(node)!=id(cls) and inspect.isfunction(cls.__dict__.get(cls.FUNCTION)):
            time_node(cls,node);_patched[node]=id(cls)


def register():
    from aiohttp import web
    from server import PromptServer
    import nodes
    server=PromptServer.instance
    if getattr(server,'_dsnn_deai_performance',False):return
    server._dsnn_deai_performance=True
    config_path=Path(__file__).with_name('targets.json')
    config=json.loads(config_path.read_text()) if config_path.is_file() else {'enabled':False}
    def on_prompt(payload):apply_nodes(nodes.NODE_CLASS_MAPPINGS,config);return payload
    server.add_on_prompt_handler(on_prompt)
    @server.routes.get('/dsnn/deai/performance')
    async def status(request):
        with _record_lock:records=list(_records)
        grouped={}
        for item in records:grouped.setdefault(item['node'],[]).append(item['seconds'])
        summary=sorted([{'node':node,'runs':len(times),'mean_seconds':round(sum(times)/len(times),6),'max_seconds':max(times)}
                        for node,times in grouped.items()],key=lambda item:item['mean_seconds'],reverse=True)
        return web.json_response({'enabled':config.get('enabled',True) and os.getenv('DSNN_DEAI_PERFORMANCE','1').lower() not in ('0','false','off'),
            'patched':sorted(_patched),'warnings':dict(_warnings),'recent':records,'summary':summary,
            'preview_edge':integer_setting('DSNN_DEAI_PREVIEW_EDGE',1024,0,8192),
            'grid_cache_bytes':_maps.bytes,'device_preference':os.getenv('DSNN_DEAI_DEVICE','auto')})
