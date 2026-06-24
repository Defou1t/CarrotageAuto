import json
from pathlib import Path


def main():
    from graphify.detect import detect
    from graphify.extract import collect_files, extract
    from graphify.cache import check_semantic_cache

    # --- Step 2: detect ---
    det = detect(Path('.'))
    Path('graphify-out/.graphify_detect.json').write_text(
        json.dumps(det, ensure_ascii=False), encoding='utf-8')
    files = det.get('files', {})
    print('DETECT total_files', det.get('total_files'), 'total_words', det.get('total_words'))
    for cat in ('code', 'document', 'paper', 'image', 'video'):
        n = len(files.get(cat, []))
        if n:
            print('  CAT', cat, n)

    # --- Step 3 Part A: AST ---
    code_files = []
    for f in files.get('code', []):
        code_files.extend(collect_files(Path(f)) if Path(f).is_dir() else [Path(f)])
    if code_files:
        result = extract(code_files, cache_root=Path('.'))
        Path('graphify-out/.graphify_ast.json').write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'AST: {len(result["nodes"])} nodes, {len(result["edges"])} edges')
    else:
        Path('graphify-out/.graphify_ast.json').write_text(
            json.dumps({'nodes': [], 'edges': [], 'input_tokens': 0, 'output_tokens': 0}, ensure_ascii=False),
            encoding='utf-8')
        print('AST: no code files')

    # --- Step 3 Part B0: semantic cache check ---
    all_files = [f for cat in ('document', 'paper', 'image') for f in files.get(cat, [])]
    cached_nodes, cached_edges, cached_hyper, uncached = check_semantic_cache(all_files)
    if cached_nodes or cached_edges or cached_hyper:
        Path('graphify-out/.graphify_cached.json').write_text(
            json.dumps({'nodes': cached_nodes, 'edges': cached_edges, 'hyperedges': cached_hyper}, ensure_ascii=False),
            encoding='utf-8')
    else:
        Path('graphify-out/.graphify_cached.json').unlink(missing_ok=True)
    Path('graphify-out/.graphify_uncached.txt').write_text('\n'.join(uncached), encoding='utf-8')
    print(f'CACHE: {len(all_files)-len(uncached)} hit, {len(uncached)} need extraction')


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    main()
