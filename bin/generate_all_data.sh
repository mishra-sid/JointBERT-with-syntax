#!/bin/bash -x

mkdir -p data/atis.bracketed.ground_truth
python generate_data.py --input_dir data/atis --output_dir data --setting bracketed.ground_truth
mkdir -p data/atis.bracketed.full.supervised
python generate_data.py --input_dir data/atis --output_dir data --setting bracketed.full.supervised
mkdir -p data/atis.bracketed.full.with_labels.supervised
python generate_data.py --input_dir data/atis --output_dir data --setting bracketed.full.with_labels.supervised
mkdir -p data/atis.bracketed.NP+VP.supervised
python generate_data.py --input_dir data/atis --output_dir data --setting bracketed.NP+VP.supervised
mkdir -p data/atis.bracketed.NP+VP.with_labels.supervised
python generate_data.py --input_dir data/atis --output_dir data --setting bracketed.NP+VP.with_labels.supervised
mkdir -p data/atis.bracketed.NP.no_nest.supervised
python generate_data.py --input_dir data/atis --output_dir data --setting bracketed.NP.no_nest.supervised
mkdir -p data/atis.bracketed.NP.supervised
python generate_data.py --input_dir data/atis --output_dir data --setting bracketed.NP.supervised
mkdir -p data/atis.bracketed.VP.supervised
python generate_data.py --input_dir data/atis --output_dir data --setting bracketed.VP.supervised
mkdir -p data/atis.control.random_50pct
python generate_data.py --input_dir data/atis --output_dir data --setting control.random_50pct
mkdir -p data/atis.control.random_50pct.bracketed.full.supervised
python generate_data.py --input_dir data/atis --output_dir data --setting control.random_50pct.bracketed.full.supervised
mkdir -p data/atis.control.less_than_avg_length
python generate_data.py --input_dir data/atis --output_dir data --setting control.less_than_avg_length
mkdir -p data/atis.control.less_than_avg_length.bracketed.full.supervised
python generate_data.py --input_dir data/atis --output_dir data --setting control.less_than_avg_length.bracketed.full.supervised
mkdir -p data/snips.bracketed.ground_truth
python generate_data.py --input_dir data/snips --output_dir data --setting bracketed.ground_truth
mkdir -p data/snips.bracketed.full.supervised
python generate_data.py --input_dir data/snips --output_dir data --setting bracketed.full.supervised
mkdir -p data/snips.bracketed.full.with_labels.supervised
python generate_data.py --input_dir data/snips --output_dir data --setting bracketed.full.with_labels.supervised
mkdir -p data/snips.bracketed.NP+VP.supervised
python generate_data.py --input_dir data/snips --output_dir data --setting bracketed.NP+VP.supervised
mkdir -p data/snips.bracketed.NP+VP.with_labels.supervised
python generate_data.py --input_dir data/snips --output_dir data --setting bracketed.NP+VP.with_labels.supervised
mkdir -p data/snips.bracketed.NP.no_nest.supervised
python generate_data.py --input_dir data/snips --output_dir data --setting bracketed.NP.no_nest.supervised
mkdir -p data/snips.bracketed.NP.supervised
python generate_data.py --input_dir data/snips --output_dir data --setting bracketed.NP.supervised
mkdir -p data/snips.bracketed.VP.supervised
python generate_data.py --input_dir data/snips --output_dir data --setting bracketed.VP.supervised
mkdir -p data/snips.control.random_50pct
python generate_data.py --input_dir data/snips --output_dir data --setting control.random_50pct
mkdir -p data/snips.control.random_50pct.bracketed.full.supervised
python generate_data.py --input_dir data/snips --output_dir data --setting control.random_50pct.bracketed.full.supervised
mkdir -p data/snips.control.less_than_avg_length
python generate_data.py --input_dir data/snips --output_dir data --setting control.less_than_avg_length
mkdir -p data/snips.control.less_than_avg_length.bracketed.full.supervised
python generate_data.py --input_dir data/snips --output_dir data --setting control.less_than_avg_length.bracketed.full.supervised