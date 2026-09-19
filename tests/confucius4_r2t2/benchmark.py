#!/usr/bin/env python3
"""Capture reproducible R2T2 warmbench reports and compare equivalent runs."""
import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import tempfile
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def signature(run):
    return (run['text'], run['language'], run['finish_delta'],
            [(c['audio_end_ms'], c['delta']) for c in run['chunks']])


def validate(report):
    if report['schema_version'] != 1 or not report['runs']:
        raise ValueError('unsupported schema or empty runs')
    if len(report['runs']) != report['config']['iterations']:
        raise ValueError('iteration count mismatch')
    for run in report['runs']:
        for value in [run['wall_ms'], run['rtf']]:
            if not math.isfinite(value) or value <= 0:
                raise ValueError('invalid timing')
        for value in [run['finish_ms']] + [c['wall_ms'] for c in run['chunks']]:
            if not math.isfinite(value) or value < 0:
                raise ValueError('invalid timing')
        if signature(run) != signature(report['runs'][0]):
            raise ValueError('output differs between iterations')


def summarize(report):
    runs = report['runs']
    chunks = [c['wall_ms'] for r in runs for c in r['chunks']]
    return dict(median_wall_ms=statistics.median(r['wall_ms'] for r in runs),
                median_rtf=statistics.median(r['rtf'] for r in runs),
                chunk_p50_ms=percentile(chunks, .5) if chunks else None,
                chunk_p95_ms=percentile(chunks, .95) if chunks else None,
                chunk_max_ms=max(chunks) if chunks else None,
                chunks_over_budget=sum(t > report['config'].get('chunk_ms', float('inf')) for t in chunks))


def compare(a, b):
    validate(a)
    validate(b)
    if a['config'] != b['config']:
        raise ValueError('benchmark configurations differ')
    for key in ('audio_sha256', 'host'):
        if a['provenance'][key] != b['provenance'][key]:
            raise ValueError(f'{key} differs')
    if signature(a['runs'][0]) != signature(b['runs'][0]):
        raise ValueError('transcript, language or chunk deltas changed')
    sa, sb = summarize(a), summarize(b)
    return dict(baseline=sa, candidate=sb,
                speedup=sa['median_wall_ms'] / sb['median_wall_ms'], output_equal=True)


def capture(args):
    binary = Path(args.binary).resolve(strict=True)
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('output exists; use a new name to preserve the baseline')
    options = args.options[1:] if args.options[:1] == ['--'] else args.options
    if '--output' in options:
        raise ValueError('pass output only to this wrapper')
    root = Path(__file__).resolve().parents[2]
    def git(*cmd):
        return subprocess.check_output(['git', '-C', str(root), *cmd], text=True).strip()
    provenance = dict(binary=str(binary), binary_sha256=sha256(binary), revision=git('rev-parse', 'HEAD'),
                      git_status=git('status', '--porcelain'),
                      diff_sha256=hashlib.sha256(subprocess.check_output(['git', '-C', str(root), 'diff', 'HEAD'])).hexdigest(),
                      host=dict(system=platform.platform(), machine=platform.machine(), node=platform.node()),
                      command=[str(binary), *options])
    with tempfile.TemporaryDirectory(prefix='r2t2-bench-') as tmp:
        raw = Path(tmp) / 'raw.json'
        subprocess.run([str(binary), *options, '--output', str(raw)], check=True)
        report = json.loads(raw.read_text())
    validate(report)
    provenance['audio_sha256'] = sha256(report['config']['audio'])
    report['provenance'] = provenance
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(summarize(report), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    run = sub.add_parser('run')
    run.add_argument('--binary', required=True)
    run.add_argument('--output', required=True)
    run.add_argument('options', nargs=argparse.REMAINDER)
    diff = sub.add_parser('compare')
    diff.add_argument('baseline', type=Path)
    diff.add_argument('candidate', type=Path)
    args = parser.parse_args()
    try:
        if args.action == 'run':
            capture(args)
        else:
            print(json.dumps(compare(json.loads(args.baseline.read_text()),
                                     json.loads(args.candidate.read_text())), indent=2))
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'benchmark: {error}\n')


if __name__ == '__main__':
    main()
