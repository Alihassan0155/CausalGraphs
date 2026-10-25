"""Separate optimized GES entry point; reference source remains unchanged."""
from pathlib import Path


def main():
    reference=Path(__file__).with_name('discover_ges.py')
    source=reference.read_text(encoding='utf-8')
    replacements={
        'from ges_recovery import build':'from ges_cpu_fast import build',
        "ROOT=b.BASE/'ges'":"ROOT=b.BASE/'ges_cpu_fast'",
        "Path(__file__),Path(__file__).with_name('ges_recovery.py')":
        "Path(__file__),Path(__file__).with_name('discover_ges.py'),Path(__file__).with_name('ges_cpu_fast.py'),Path(__file__).with_name('ges_recovery.py')"}
    for old,new in replacements.items():
        assert source.count(old)==1,'Reference wrapper changed; review launcher'
        source=source.replace(old,new)
    exec(compile(source,str(reference),'exec'),dict(__name__='__main__',__file__=str(Path(__file__).resolve())))


if __name__=='__main__':main()
