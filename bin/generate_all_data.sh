#!/bin/bash -x

#python generate_data.py --input_dir data/atis --output_dir data --setting bracketed.ground_truth
#python generate_data.py --input_dir data/atis --output_dir data --setting bracketed.full.supervised
python generate_data.py --input_dir data/atis --output_dir data --setting bracketed.NP+VP.supervised
python generate_data.py --input_dir data/atis --output_dir data --setting bracketed.NP.no_nest.supervised
python generate_data.py --input_dir data/atis --output_dir data --setting bracketed.NP.supervised
python generate_data.py --input_dir data/atis --output_dir data --setting bracketed.VP.supervised

python generate_data.py --input_dir data/snips --output_dir data --setting bracketed.ground_truth
python generate_data.py --input_dir data/snips --output_dir data --setting bracketed.full.supervised
python generate_data.py --input_dir data/snips --output_dir data --setting bracketed.NP+VP.supervised
python generate_data.py --input_dir data/snips --output_dir data --setting bracketed.NP.no_nest.supervised
python generate_data.py --input_dir data/snips --output_dir data --setting bracketed.NP.supervised
python generate_data.py --input_dir data/snips --output_dir data --setting bracketed.VP.supervised