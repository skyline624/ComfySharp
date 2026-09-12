"""Opt-in reference execution of frozen ComfyUI bodies with the shared SD1.5 checkpoint.
No product outputs, model copies, network access or synthetic replacement networks.
Python is confined to this development laboratory; .NET tests never import it.
"""
import argparse, hashlib, importlib.util, json, logging, math, os, struct, sys, time, types
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]

def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()

def tokens(text, tokenizer, protocol):
    # Independent fixture input recipe for this fixed lowercase ASCII sentence only.
    # This is not a replacement for ComfyUI's general weighted/tokenizer behavior.
    for name,pin in protocol['tokenizerFiles'].items():
        assert digest(tokenizer/name)==pin
    vocabulary=json.loads((tokenizer/'vocab.json').read_text(encoding='utf-8'))
    ranks={tuple(line.split()):i for i,line in enumerate((tokenizer/'merges.txt').read_text(encoding='utf-8').splitlines()[1:]) if line.strip()}
    ids=[49406]
    for word in text.split():
        assert word.isascii() and word.isalpha() and word.islower()
        pieces=list(word[:-1])+[word[-1]+'</w>']
        while len(pieces)>1:
            pairs=list(zip(pieces,pieces[1:]))
            pair=min(pairs,key=lambda p:ranks.get(p,float('inf')))
            if pair not in ranks: break
            merged=[]; i=0
            while i<len(pieces):
                if i+1<len(pieces) and (pieces[i],pieces[i+1])==pair:
                    merged.append(pieces[i]+pieces[i+1]); i+=2
                else: merged.append(pieces[i]); i+=1
            pieces=merged
        ids.extend(vocabulary[p] for p in pieces)
    assert len(ids)<77
    return [[(i,1.0) for i in ids+[49407]*(77-len(ids))]]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.source.resolve(strict=True); checkpoint=args.checkpoint.resolve(strict=True)
    output=args.output.resolve()
    assert not output.exists() and output.parent.is_dir()
    for protected in (source,checkpoint,args.tokenizer.resolve(strict=True)):
        assert not output.is_relative_to(protected) and not protected.is_relative_to(output)
    protocol=json.loads((HERE/'protocol.json').read_text())
    paths=list(HERE.glob('*.py'))+[HERE/'protocol.json']
    helper_names=['labs/sd15-pipeline-source/reference.py','labs/clip-source/reference.py','labs/sd-source/common.py','labs/sd-source/unet.py','labs/sd-source/vae.py']
    paths += [REPO/name for name in helper_names]
    before={str(p.relative_to(REPO)):digest(p) for p in paths}
    assert checkpoint.stat().st_size==protocol['modelBytes'] and digest(checkpoint)==protocol['modelSha256']
    blobs={entry['file']:(source/entry['file']).read_bytes() for entry in protocol['sourceFiles']}
    for entry in protocol['sourceFiles']:
        assert hashlib.sha256(blobs[entry['file']]).hexdigest()==entry['sha256']
    assert digest(source/'comfy/sd1_clip_config.json')==protocol['clipConfigSha256']
    positive_tokens=tokens(protocol['case']['prompt'],args.tokenizer,protocol)
    negative_tokens=tokens(protocol['case']['negative'],args.tokenizer,protocol)
    helper=load(REPO/'labs/sd15-pipeline-source/reference.py','pretrained_pipeline_helper')
    model_adapter=load(HERE/'models.py','pretrained_model_adapter'); model_adapter.bind(helper)
    evidence=[]; started=time.monotonic()
    output.mkdir()
    def stage(name): print(f'{time.monotonic()-started:.1f}s {name}',flush=True)
    stage('construct')
    torch,np,common,models,clip,vae=model_adapter.build_models(REPO,source,protocol,blobs,evidence)
    torch.set_num_threads(protocol['case']['threads'])
    prefixes={'clip':'cond_stage_model.transformer.','unet':'model.diffusion_model.','vae':'first_stage_model.'}
    loaded=[]
    stage('load')
    with checkpoint.open('rb') as stream:
        header_size=struct.unpack('<Q',stream.read(8))[0]
        assert header_size<=16*1024*1024
        header=json.loads(stream.read(header_size))
        with torch.no_grad():
            for component,model in models.items():
                for name,parameter in model.named_parameters():
                    if component=='clip' and name=='text_projection.weight':
                        # Original source computes this auxiliary projection. SD1 uses the
                        # unprojected pooled result and hidden output; neither uses this matrix.
                        parameter.copy_(torch.eye(*parameter.shape)); continue
                    key=prefixes[component]+name
                    item=header[key]
                    assert item['dtype']=='F32' and item['shape']==list(parameter.shape),(key,item['shape'],list(parameter.shape))
                    begin,end=item['data_offsets']
                    assert 0<=begin<end and header_size+8+end<=protocol['modelBytes']
                    stream.seek(8+header_size+begin)
                    raw=stream.read(end-begin)
                    assert len(raw)==parameter.numel()*4
                    parameter.copy_(torch.from_numpy(np.frombuffer(raw,dtype='<f4').copy().reshape(item['shape'])))
                    pin=hashlib.sha256(raw).hexdigest()
                    assert helper.tensor_hash(parameter)==pin
                    loaded.append({'source':key,'shape':item['shape'],'sha256':pin})
    records=[]
    def capture(name,value):
        assert value.device.type=='cpu' and value.dtype==torch.float32 and value.isfinite().all().item()
        raw=value.detach().contiguous().numpy().tobytes()
        filename=name+'.f32'
        with (output/filename).open('xb') as file: file.write(raw)
        records.append({'File':filename,'Shape':list(value.shape),'Dtype':'float32-le','Bytes':len(raw),'Sha256':hashlib.sha256(raw).hexdigest()})
    namespace={'torch':torch,'np':np,'math':math,'logging':logging}
    def original(relative,names,target=None):
        helper.load_source(blobs,protocol,relative,names,namespace if target is None else target,evidence)
    original('comfy/ldm/modules/diffusionmodules/util.py',['make_beta_schedule'])
    original('comfy/model_sampling.py',['reshape_sigma','EPS','ModelSamplingDiscrete'])
    original('comfy/samplers.py',['cfg_function','sampling_function','Sampler'])
    utility={}; original('comfy/k_diffusion/utils.py',['append_dims'],utility)
    namespace['utils']=types.SimpleNamespace(append_dims=utility['append_dims'])
    namespace['trange']=lambda count,disable=None:range(count)
    original('comfy/k_diffusion/sampling.py',['append_zero','get_sigmas_karras','to_d','sample_euler'])
    original('comfy/latent_formats.py',['LatentFormat','SD15'])
    with torch.no_grad():
        stage('encode')
        positive,_=clip.encode_token_weights(positive_tokens)
        negative,_=clip.encode_token_weights(negative_tokens)
        capture('positive-hidden',positive); capture('negative-hidden',negative)
        case=protocol['case']
        generator=torch.Generator(device='cpu').manual_seed(case['seed'])
        noise=torch.randn([1,4,case['height']//8,case['width']//8],generator=generator,dtype=torch.float32)
        schedule=namespace['ModelSamplingDiscrete']()
        sigmas=namespace['get_sigmas_karras'](case['steps'],float(schedule.sigma_min),float(schedule.sigma_max))
        capture('noise',noise); capture('sigmas',sigmas)
        eps=namespace['EPS'](); eps.sigma_data=1.0
        maximum=namespace['Sampler']().max_denoise(types.SimpleNamespace(inner_model=types.SimpleNamespace(model_sampling=schedule)),sigmas)
        assert maximum
        initial=eps.noise_scaling(sigmas[0],noise,torch.zeros_like(noise),maximum)
        capture('initial',initial)
        def denoise(x,sigma,context):
            scaled=eps.calculate_input(sigma,x)
            timestep=schedule.timestep(sigma).float().reshape(-1)
            raw=models['unet']._forward(scaled,timesteps=timestep,context=context,y=None,control=None,transformer_options={})
            return eps.calculate_denoised(sigma,raw,x)
        def calc(args):
            cond,uncond=args['conds']; x,sigma=args['input'],args['sigma']
            return [denoise(x,sigma,cond),torch.zeros_like(x) if uncond is None else denoise(x,sigma,uncond)]
        def guided(x,sigma):
            return namespace['sampling_function'](None,x,sigma,negative,positive,case['cfg'],model_options={'disable_cfg1_optimization':False,'sampler_calc_cond_batch_function':calc})
        stage('sample')
        final=namespace['sample_euler'](guided,initial,sigmas,disable=True,s_churn=0.0)
        final=eps.inverse_noise_scaling(sigmas[-1],final)
        capture('diffusion',final)
        raw=namespace['SD15'](scale_factor=0.18215).process_out(final)
        capture('raw-vae',raw)
        stage('decode')
        image=vae.decode(raw)
        capture('image',image)
    for component,model in models.items():
        for name,parameter in model.named_parameters():
            key=prefixes[component]+name
            row=next((r for r in loaded if r['source']==key),None)
            if row is not None: assert helper.tensor_hash(parameter)==row['sha256']
    assert before=={str(p.relative_to(REPO)):digest(p) for p in paths}
    assert checkpoint.stat().st_size==protocol['modelBytes'] and digest(checkpoint)==protocol['modelSha256']
    assert all(hashlib.sha256((source/name).read_bytes()).hexdigest()==hashlib.sha256(raw).hexdigest() for name,raw in blobs.items())
    report={'status':'ok','familyQualified':False,'profile':protocol['profile'],'modelSha256':protocol['modelSha256'],
        'case':case,'positiveTokens':positive_tokens,'negativeTokens':negative_tokens,'traces':records,
        'loadedParameters':len(loaded),'loadedParametersManifestSha256':hashlib.sha256(helper.compact(loaded)).hexdigest(),
        'auxiliaryProjection':'Identity, absent from checkpoint; source computes unused projected pool only.',
        'sourceBodies':evidence,'sourceFilesUnchanged':True,'checkpointUnchanged':True,'parametersUnchanged':True,'laboratoryHashes':before,
        'runtime':{'torch':torch.__version__,'numpy':np.__version__,'threads':torch.get_num_threads(),'interopThreads':torch.get_num_interop_threads()},
        'elapsedSeconds':time.monotonic()-started}
    with (output/'result.json').open('x') as file: json.dump(report,file,indent=2)
    stage('complete')

if __name__=='__main__': main()
