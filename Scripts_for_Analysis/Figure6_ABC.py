"""Generate the final Figure 6A, 6B, and 6C panels from saved numerical inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=CODE_ROOT / 'tables')
    parser.add_argument('--output-dir', type=Path, default=CODE_ROOT / 'figures')
    parser.add_argument('--panels', nargs='+', choices=['A', 'B', 'C'], default=['A', 'B', 'C'])
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    from xenium_pipeline.model_comparison import plot_model_comparison
    from xenium_pipeline.feature_importance_comparison import plot_figure7_scatter, FIGURE7_ENCODERS
    from xenium_pipeline.figure_6c import main as plot_survival

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    for panel in dict.fromkeys(args.panels):
        if panel == 'A':
            source = args.data_dir / 'figure_6a_fold_metrics.csv'
            metrics = pd.read_csv(source, float_precision='round_trip')
            figure, _, _, matched = plot_model_comparison(metrics, args.output_dir)
            matched.to_csv(args.output_dir / 'Figure_6A_fold_metrics.csv', index=False)
            plt.close(figure)
        elif panel == 'B':
            source = args.data_dir / 'figure_6b_feature_profiles.csv.gz'
            profiles = pd.read_csv(source, float_precision='round_trip')
            correlations = plot_figure7_scatter(profiles, args.output_dir, encoders=FIGURE7_ENCODERS,
                views=['coad_visium_vs_brca_xenium'], manuscript_panel=True)
            correlations.to_csv(args.output_dir / 'Figure_6B_correlations.csv', index=False)
        else:
            source = args.data_dir / 'figure_6c_metadata.json'
            plot_survival(['--data-dir', str(args.data_dir), '--output-dir', str(args.output_dir)])
        sources.append(dict(panel=panel, input=str(source.resolve()),
            sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    (args.output_dir / 'Figure_6_provenance.json').write_text(json.dumps(sources, indent=2) + '\n')


if __name__ == '__main__':
    main()
