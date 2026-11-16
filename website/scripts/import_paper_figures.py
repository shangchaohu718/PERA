"""Extract original paper figures for the local research website, without redrawing data."""
import argparse
import hashlib
import json
from pathlib import Path
from pypdf import PdfReader

WEBSITE = Path(__file__).resolve().parents[1]
FIGURES = [
    {'id': 'transfer', 'figure': 3, 'page': 7, 'image': 'Im3.png',
     'file': 'unseen-dance-transfer.png', 'size': [4320, 1512],
     'caption': 'Execution of dance motions not used to train the expert. (a) Fraction of episodes that fully execute the reference motion. (b) Mean motion completion. (c) One replay of the motion with the largest completion improvement: joint-angle and aligned-body-position RMSE are compared over the first 66 frames during which neither policy has terminated. Base-policy error is normalized to 100%; blue bars show the relative error with the expert enabled.'},
    {'id': 'teacher-comparison', 'figure': 5, 'page': 8, 'image': 'Im5.png',
     'file': 'expert-teacher-errors.png', 'size': [4320, 1596],
     'caption': 'Position errors of HoloMotion + expert and task teachers. The comparison includes the five tasks on which both policies fully complete all 16 evaluation episodes. (a) Per-task position-error reductions relative to the teacher: positive values indicate lower error and negative values indicate higher error than the teacher. (b) Position errors averaged equally across the five tasks.'},
]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paper', type=Path)
    args = parser.parse_args()
    source = args.paper.resolve()
    reader = PdfReader(source)
    output = WEBSITE / 'public' / 'figures' / 'paper'
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'source_filename': source.name,
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'source_title': str(reader.metadata.title),
                'imported_on': '2026-11-13',
                'transformation': 'Original embedded PNG images extracted with pypdf. No resampling, redrawing, recoloring, cropping or numerical changes. Full paper is not served.',
                'figures': []}
    for spec in FIGURES:
        matches = [im for im in reader.pages[spec['page'] - 1].images if im.name == spec['image']]
        if len(matches) != 1:
            raise ValueError(f"Cannot identify original figure {spec['figure']}")
        original = matches[0]
        if list(original.image.size) != spec['size']:
            raise ValueError(f"Unexpected dimensions for figure {spec['figure']}")
        target = output / spec['file']
        target.write_bytes(original.data)
        manifest['figures'].append({**spec, 'sha256': hashlib.sha256(original.data).hexdigest()})
        print(f"Figure {spec['figure']}: {target.relative_to(WEBSITE)}")
    (output / 'provenance.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')

if __name__ == '__main__':
    main()
