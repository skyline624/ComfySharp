"""Prospective native input allocation and stdlib record validator.

Imports no native runtime. Torch is supplied by the separately guarded collector.
"""
import hashlib

POLICY = 'sd-native-aligned-inputs-v1'
ORDER = {'unet': ('latent','timesteps','context'), 'guidance': ('latent','sigma','positive','negative')}

def require(value, message):
    if not value: raise ValueError(message)

def layout(value):
    require(str(value.dtype)=='torch.float32' and value.device.type=='cpu' and not value.is_sparse,
            'Only dense CPU/F32 inputs.')
    require(value.is_contiguous() and value.numel()>0, 'Unexpected noncontiguous/empty input.')
    residue = value.data_ptr() % 64
    return {'dtype':'float32','shape':list(value.shape),'stride':list(value.stride()),
            'storageOffset':value.storage_offset(),'addressModulo64':residue,'aligned64':residue==0,
            'sha256':hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()}

def identity(record):
    return tuple(record[k] if k not in ('shape','stride') else tuple(record[k])
                 for k in ('dtype','shape','stride','storageOffset','sha256'))

class Prepared:
    def __init__(self, torch, kind, borrowed):
        require(tuple(borrowed)==ORDER[kind], 'Input preparation order differs.')
        self.originals=borrowed;self.original_layouts={};self.values={};self.observations=[]
        for name, value in borrowed.items():
            before=layout(value)
            require(before['storageOffset']==0, 'This prospective corpus admits only zero storage offsets.')
            copy=torch.empty(tuple(value.shape),dtype=torch.float32,device='cpu')
            copy.copy_(value)
            after=layout(copy)
            require(identity(before)==identity(after) and after['addressModulo64']==0,'Prepared input bytes/layout differ.')
            require(copy.data_ptr()!=value.data_ptr() and all(copy.data_ptr()!=v.data_ptr() for v in self.values.values()),
                    'Prepared inputs must own independent buffers.')
            self.original_layouts[name]=before;self.values[name]=copy

    def observe(self):
        snapshot={}
        for name, value in self.values.items():
            record=layout(value)
            require(identity(record)==identity(self.original_layouts[name]) and record['addressModulo64']==0,
                    'Actual prepared input changed.')
            require(identity(layout(self.originals[name]))==identity(self.original_layouts[name]),'Borrowed input changed.')
            snapshot[name]=record
        self.observations.append(snapshot)

    def evidence(self):
        require(len(self.observations)==6,'Three before/after observations required.')
        return {'policy':POLICY,'preparationOrder':list(self.values),'allocation':'nativeEmptyThenCopy',
                'originalLayouts':self.original_layouts,'observations':self.observations,
                'observationOrder':['before0','after0','before1','after1','before2','after2'],
                'independentBuffers':True,'originalInputsUnchanged':True}

def validate(doc):
    """Only input metadata is checked here; the pinned comparator validates payloads."""
    evidence=doc['inputAllocation'];order=ORDER[doc['kind']]
    require(evidence['policy']==POLICY and evidence['allocation']=='nativeEmptyThenCopy','Wrong allocation policy.')
    require(evidence['preparationOrder']==list(order),'Preparation order differs.')
    require(evidence['independentBuffers'] is True and evidence['originalInputsUnchanged'] is True,'Ownership/integrity missing.')
    require(evidence['observationOrder']==['before0','after0','before1','after1','before2','after2'],'Wrong observations.')
    require(len(evidence['observations'])==6 and set(evidence['originalLayouts'])==set(order),'Incomplete allocation evidence.')
    for snapshot in evidence['observations']:
        require(set(snapshot)==set(order),'Actual input missing.')
        for name, record in snapshot.items():
            expected=doc['inputs'][name];original=evidence['originalLayouts'][name]
            require(record['dtype']=='float32' and record['storageOffset']==0 and record['addressModulo64']==0 and record['aligned64'] is True,
                    'Actual input not zero-offset aligned CPU/F32.')
            require(identity(record)==identity(original),'Original/prepared bytes or layout differ.')
            require(all(record[key]==expected[key] for key in ('dtype','shape','stride','sha256','storageOffset','addressModulo64','aligned64')),
                    'Evidence does not describe recorded actual inputs.')
