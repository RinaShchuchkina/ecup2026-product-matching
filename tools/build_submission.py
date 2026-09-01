import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORBIDDEN = [r'/home/[a-z]', r'/Users/[a-z]', r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
             r'ssh -', r'sbatch', r'squeue', r'BEGIN [A-Z ]*PRIVATE KEY']
MAX_BYTES = 5 * 1024 ** 3
RUNTIME = ['run.sh', 'run.py', 'serialize.py', 'metadata.json']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stage', default='')
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding='utf-8'))
    ckpt = os.path.join(ROOT, 'weights', cfg['checkpoint'])
    if not os.path.isdir(ckpt):
        sys.exit(f'нет каталога весов: {ckpt}')
    stage = args.stage or os.path.join(ROOT, 'build', os.path.splitext(os.path.basename(args.config))[0])
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    os.makedirs(os.path.join(stage, 'model2'))

    for name in RUNTIME:
        shutil.copy2(os.path.join(ROOT, 'src', name), os.path.join(stage, name))
    shutil.copytree(os.path.join(ROOT, 'src', 'wheels'), os.path.join(stage, 'wheels'))
    for f in sorted(os.listdir(ckpt)):
        shutil.copy2(os.path.join(ckpt, f), os.path.join(stage, 'model2', f))

    run_py = os.path.join(stage, 'run.py')
    src = open(run_py, encoding='utf-8').read()
    src = re.sub(r"^CE = .*$", "CE = [('model2', 1.0, %d)]" % int(cfg.get('batch', 384)), src, count=1, flags=re.M)
    src = re.sub(r"^MAXLEN = .*$", "MAXLEN = %d" % int(cfg.get('max_length', 384)), src, count=1, flags=re.M)
    src = re.sub(r"^CONFIG = .*$", "CONFIG = ''", src, count=1, flags=re.M)
    src = re.sub(r"^WEIGHTS_ROOT = .*$", "WEIGHTS_ROOT = HERE", src, count=1, flags=re.M)
    src = src.replace("    cap = 500\n", "    cap = %d\n" % int(cfg.get('max_attr_chars', 500)), 1)
    src = src.replace("    use_tta = True\n", "    use_tta = %r\n" % bool(cfg.get('swap_tta', True)), 1)
    open(run_py, 'w', encoding='utf-8').write(src)
    compile(src, run_py, 'exec')

    meta = os.path.join(stage, 'metadata.json')
    m = json.load(open(meta, encoding='utf-8'))
    if m.get('entry_point') != 'bash run.sh':
        sys.exit('metadata.json: entry_point должен быть "bash run.sh"')

    problems = []
    for base, _, files in os.walk(stage):
        for f in files:
            p = os.path.join(base, f)
            rel = os.path.relpath(p, stage)
            if not rel.isascii():
                problems.append(f'не-ASCII в имени: {rel}')
            if os.path.splitext(f)[1] in ('.py', '.sh', '.json'):
                text = open(p, encoding='utf-8', errors='ignore').read()
                for pat in FORBIDDEN:
                    hit = re.search(pat, text)
                    if hit:
                        problems.append(f'{rel}: {hit.group(0)}')
    total = sum(os.path.getsize(os.path.join(b, f)) for b, _, fs in os.walk(stage) for f in fs)
    if total > MAX_BYTES:
        problems.append(f'распакованный размер {total} > {MAX_BYTES}')
    if not os.path.isfile(os.path.join(stage, 'metadata.json')):
        problems.append('metadata.json не в корне')
    if problems:
        for x in problems:
            print('ОТКАЗ:', x)
        sys.exit(1)

    if os.path.exists(args.out):
        os.remove(args.out)
    with zipfile.ZipFile(args.out, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for base, _, files in os.walk(stage):
            for f in sorted(files):
                p = os.path.join(base, f)
                z.write(p, os.path.relpath(p, stage))
    with zipfile.ZipFile(args.out) as z:
        bad = z.testzip()
        if bad:
            sys.exit(f'битый архив: {bad}')
        names = z.namelist()
    for need in RUNTIME + ['model2/model.safetensors']:
        if need not in names:
            sys.exit(f'в архиве нет {need}')
    print(f'собрано {args.out}')
    print(f'  файлов {len(names)}, архив {os.path.getsize(args.out) / 1e6:.1f} МБ, распакованный {total / 1e6:.1f} МБ')
    print(f'  чекпоинт {cfg["checkpoint"]}, cap {cfg.get("max_attr_chars", 500)}, maxlen {cfg.get("max_length", 384)}, tta {cfg.get("swap_tta", True)}')
    print(f'  sha256 весов {sha256(os.path.join(stage, "model2", "model.safetensors"))}')
    print(f'  sha256 архива {sha256(args.out)}')


main()
