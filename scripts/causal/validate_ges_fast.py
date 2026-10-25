"""Reuse the graph validator against separate optimized-CPU artifacts."""
from pathlib import Path


if __name__=='__main__':
    path=Path(__file__).with_name('validate_ges.py')
    source=path.read_text(encoding='utf-8')
    anchor="folder=b.BASE/'ges'/args.run_id"
    assert source.count(anchor)==1
    exec(compile(source.replace(anchor,"folder=b.BASE/'ges_cpu_fast'/args.run_id"),str(path),'exec'),
         dict(__name__='__main__',__file__=str(path)))
