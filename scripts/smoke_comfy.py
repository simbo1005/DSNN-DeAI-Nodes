"""Exercise all four custom nodes with a synthetic image; no model downloads."""
import argparse
import json
from pathlib import Path
import sys
import time
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import install


def request(url,payload=None):
    body=json.dumps(payload).encode() if payload is not None else None
    with urlopen(Request(url,data=body,headers={'Content-Type':'application/json'}),timeout=30) as response:
        return json.load(response)


def inputs(schema,image_link):
    values={}
    for name,spec in schema['input']['required'].items():
        kind=spec[0];options=spec[1] if len(spec)>1 and isinstance(spec[1],dict) else {}
        if kind=='IMAGE':values[name]=image_link
        elif isinstance(kind,list):values[name]=options.get('default',kind[0])
        elif 'default' in options:values[name]=options['default']
        elif kind in ('INT','FLOAT'):values[name]=options.get('min',0)
        elif kind=='BOOLEAN':values[name]=False
        elif kind=='STRING':values[name]=''
        else:raise RuntimeError(f'Unhandled smoke input {name}: {kind}')
    return values


def main():
    import numpy as np
    from PIL import Image,JpegImagePlugin
    parser=argparse.ArgumentParser();parser.add_argument('--comfyui',type=Path,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:8188')
    parser.add_argument('--label',choices=('baseline','optimized'),default='optimized');options=parser.parse_args()
    root=options.comfyui;url=options.url.rstrip('/')
    install.verify(url,install.load_lock())
    folder=root/'input';folder.mkdir(exist_ok=True)
    x=np.linspace(0,255,128,dtype=np.uint8)
    image=np.stack(np.broadcast_arrays(x[None,:],x[:,None],np.full((128,128),117,dtype=np.uint8)),axis=-1)
    Image.fromarray(image).save(folder/'dsnn-deai-synthetic.png')
    registry=request(url+'/object_info')
    graph={'1':{'class_type':'LoadImage','inputs':{'image':'dsnn-deai-synthetic.png'}}}
    previous=['1',0]
    for index,node in enumerate(('Vintage Effect','LCLensProfile','RadianceCompressionArtifacts','BondSaveWithCustomMetadata'),2):
        value=inputs(registry[node],previous)
        if node=='Vintage Effect':value.update(grain_strength=0)
        if node=='RadianceCompressionArtifacts':value.update(block_size=1,quality=92,color_subsampling=False,noise_amount=0,seed=42)
        if node=='BondSaveWithCustomMetadata':
            # Keep the test entirely local (no geocoding or source metadata).
            value.update(filename_prefix='dsnn-deai-smoke-'+options.label+'/result',location_city='',source_filepath='',artist='DSNN automated test',custom_xmp_tags='')
        graph[str(index)]={'class_type':node,'inputs':value};previous=[str(index),0]
    result=request(url+'/prompt',{'prompt':graph,'client_id':'dsnn-deai-ci'})
    prompt_id=result['prompt_id']
    for _ in range(120):
        history=request(url+'/history/'+prompt_id)
        if prompt_id in history:break
        time.sleep(1)
    else:raise RuntimeError('Synthetic De-AI execution did not complete')
    job=history[prompt_id]
    if job.get('status',{}).get('status_str')!='success':raise RuntimeError(json.dumps(job.get('status',{}),indent=2))
    saved=list((root/('output/dsnn-deai-smoke-'+options.label)).glob('*.jpg'))
    if len(saved)!=1:raise RuntimeError('Bond did not save exactly one JPEG')
    with Image.open(saved[0]) as output:
        assert output.format=='JPEG' and output.mode=='RGB' and output.size==(128,128)
        assert JpegImagePlugin.get_sampling(output)==0,'JPEG must retain the uploaded 4:4:4 setting'
        pixels=np.array(output)
    runtime=request(url+'/dsnn/deai/performance')
    if options.label=='baseline':
        assert runtime['enabled'] is False and runtime['patched']==[]
    else:
        assert runtime['enabled'] is True and not runtime['warnings'],runtime
        assert {'Vintage Effect','LCLensProfile','RadianceCompressionArtifacts','BondSaveWithCustomMetadata'}<=set(runtime['patched'])
        bond=next(r for r in reversed(runtime['recent']) if r['node']=='BondSaveWithCustomMetadata')
        assert bond['metadata_reads']==1,bond
        baseline_file=next((root/'output/dsnn-deai-smoke-baseline').glob('*.jpg'))
        with Image.open(baseline_file) as original:expected=np.array(original)
        maximum=int(np.abs(pixels.astype(int)-expected.astype(int)).max())
        assert maximum<=1,f'Optimized CPU output differs by {maximum} code values'
        print('Baseline comparison: max decoded JPEG channel difference =',maximum)
        print('Performance:',json.dumps(runtime['summary'],indent=2))
    print('PASS: all 10 node types registered; Vintage → LC Lens → Radiance block_size=1 → patched Bond JPEG executed successfully.')


if __name__=='__main__':main()
