"""Pure filesystem controls for disposable Linux native-origin copies."""
import hashlib
import os
from pathlib import Path
import shutil

CORE = ('libc10.so', 'libtorch_cpu.so', 'libtorch.so')
BINDING = 'libLibTorchSharp.so'
SYSTEM = {'libdl.so.2', 'libstdc++.so.6', 'libm.so.6', 'libgcc_s.so.1',
          'libpthread.so.0', 'libc.so.6', 'ld-linux-x86-64.so.2', 'librt.so.1'}


def require(value, message):
    if not value: raise ValueError(message)


def digest(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def disjoint(a, b):
    return not a.is_relative_to(b) and not b.is_relative_to(a)


def protect_destinations(output, staging, protected):
    destinations = []
    for path in (output, staging):
        require(path.is_absolute(), 'Destination must be absolute.')
        resolved = path.resolve()
        require(not path.exists() and not path.is_symlink(), 'Destination already exists.')
        require(all(disjoint(resolved, p.resolve(strict=True)) for p in protected),
                'Destination overlaps a protected input.')
        destinations.append(resolved)
    require(disjoint(*destinations), 'Evidence and disposable copies must not overlap.')
    return tuple(destinations)


def inventory(root):
    """Never follow symlinks or accept non-regular payloads; names are relative only."""
    require(root.is_dir() and not root.is_symlink(), 'Inventory root must be a regular directory.')
    files, directories = {}, []
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(), 'Symlink in inventory: '+path.relative_to(root).as_posix())
        name = path.relative_to(root).as_posix()
        if path.is_dir(): directories.append(name)
        else:
            require(path.is_file(), 'Non-regular inventory entry: '+name)
            files[name] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    return {'directories': directories, 'files': files}


def probe_paths(before):
    files = before['files']
    probes = {name: [p for p in files if Path(p).name == name] for name in (*CORE, BINDING)}
    for name, paths in probes.items():
        require(paths, 'Native probe is missing: '+name)
        require(len({(files[p]['bytes'],files[p]['sha256']) for p in paths}) == 1,
                'Conflicting native probe identities: '+name)
    openmp = [p for p in files if Path(p).name.startswith(('libgomp', 'libiomp', 'libomp'))]
    require(openmp and all(Path(p).name.startswith('libgomp') for p in openmp), 'Expected only GNU OpenMP probes.')
    require(len({(files[p]['bytes'],files[p]['sha256']) for p in openmp}) == 1, 'Conflicting OpenMP probe identities.')
    # Unexpected versioned core candidates would defeat a complete substitution.
    selected = {p for paths in probes.values() for p in paths} | set(openmp)
    for p in files:
        name = Path(p).name
        require(not any(name.startswith(n+'.') for n in (*CORE,BINDING)) or p in selected,
                'Unclassified versioned native probe: '+p)
    return probes, openmp


def validate_elf(core, binding, openmp):
    require(set(core) == set(CORE), 'Core ELF set incomplete.')
    require(all(core[n]['soname'] == n for n in CORE), 'Core SONAME differs.')
    require(openmp['soname'] == 'libgomp.so.1', 'OpenMP SONAME differs.')
    require(binding['soname'] in (None, BINDING), 'Binding SONAME differs.')
    require(set(CORE).issubset(binding['needed']), 'Binding does not declare all three core dependencies.')
    allowed = SYSTEM | set(CORE) | {'libgomp.so.1', openmp['name']}
    for record in (*core.values(),binding,openmp):
        require(set(record['needed']).issubset(allowed), 'Unclosed ELF dependency: '+record['name'])
    require('libgomp.so.1' in core['libtorch_cpu.so']['needed'] or
            openmp['name'] in core['libtorch_cpu.so']['needed'], 'CPU OpenMP dependency is missing.')


def copy_original(original, target, before):
    require(not target.exists() and disjoint(original.resolve(),target.resolve()), 'Unsafe copy target.')
    shutil.copytree(original, target, copy_function=shutil.copy2)
    require(inventory(target) == before, 'Copy inventory differs from original.')
    require(all(not os.path.samefile(original/p,target/p) for p in before['files']),
            'Copy shares a payload inode with original.')


def wheel_substitution(target, before, probes, openmp_paths, wheel, wheel_core, wheel_openmp):
    """Replace only copied payloads. All aliases refer to internal, copied inodes."""
    expected = {'directories': list(before['directories']), 'files': dict(before['files'])}
    aliases = {}
    groups = {n: list(probes[n]) for n in CORE}
    native_dirs = {str(Path(p).parent) for n in (*CORE,BINDING) for p in probes[n]}
    groups['libgomp.so.1'] = sorted(set(openmp_paths) | {str(Path(d)/'libgomp.so.1').replace('\\','/') for d in native_dirs})
    for name, paths in groups.items():
        record = wheel_openmp if name == 'libgomp.so.1' else wheel_core[name]
        source = wheel/record['name']
        require(source.is_file() and digest(source) == record['sha256'] and source.stat().st_size == record['bytes'],
                'Wheel input identity changed.')
        anchor = target/paths[0]
        for relative in paths:
            destination = target/relative
            require(destination.resolve().is_relative_to(target.resolve()) and not destination.is_symlink(),
                    'Unsafe substitution target.')
            if destination.exists():
                require(relative in before['files'], 'Uninventoried existing substitution target.')
                destination.unlink()
        shutil.copy2(source, anchor)
        require(not os.path.samefile(source,anchor), 'Copied wheel payload shares package inode.')
        for relative in paths[1:]: os.link(anchor,target/relative)
        require(all(os.path.samefile(anchor,target/p) for p in paths), 'Native aliases must share one inode.')
        for relative in paths: expected['files'][relative] = {'bytes':record['bytes'],'sha256':record['sha256']}
        aliases[name] = {'relativePaths':paths,'sameInode':True,'independentOfPackage':True}
    require(inventory(target) == expected, 'Wheel copy contains an unexpected modification.')
    return expected, aliases


def verify_aliases(target, aliases):
    for group in aliases.values():
        paths = [target/p for p in group['relativePaths']]
        require(all(p.is_file() and not p.is_symlink() and os.path.samefile(paths[0],p) for p in paths),
                'Copied native alias no longer shares the selected inode.')
    return True
