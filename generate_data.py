import argparse
from pathlib import Path
import shutil

import nltk
import benepar
from tqdm import tqdm

parser = benepar.Parser("benepar_en3")

def get_parse_tree_brackets(tree, allowed_all_constituencies=False, allowed_constituencies=[], allowed_nesting=True):
    sent = []
    for node in tree:
        if type(node) is nltk.Tree:
            inner = get_parse_tree_brackets(node, allowed_all_constituencies, allowed_constituencies, allowed_nesting)
            if node.label() != 'S' and (allowed_all_constituencies or node.label() in allowed_constituencies) and \
                    (allowed_nesting or ( '-LBRAC-' not in inner and '-RBRAC-' not in inner )):
                sent.append('-LBRAC-')
                sent.extend(inner)
                sent.append('-RBRAC-')
            else:
                sent.extend(inner)
        else:
            sent.append(node)
    return sent

def get_ground_truth_brackets(inp_words, target_slots):
    b_words = []

    c_state = 'O'
    for word, slot in zip(inp_words, target_slots):
        if slot.startswith('B-'):
            if c_state == 'O':
                b_words.append('-LBRAC-')
            else:
                b_words.extend(['-RBRAC-', '-LBRAC-'])
            b_words.append(word)
            c_state = 'B'
        elif slot.startswith('I-'):
            b_words.append(word)
            c_state = 'I'
        else: # O
            if c_state != 'O':
                b_words.append('-RBRAC-')
            b_words.append(word)
            c_state = 'O'
    
    if c_state != 'O':
        b_words.append('-RBRAC-')
    
    return b_words

def write_data(inp_path, out_path, setting):
    if setting.startswith("bracketed"):
        shutil.copy(inp_path / "label", out_path)

        with open(inp_path / "seq.in") as inp_inp, open(inp_path / "seq.out") as inp_out, \
                open(out_path / "seq.in", "w") as out_in, open(out_path / "seq.out", "w") as out_out:
            for sent_inp, sent_out in tqdm(zip(inp_inp, inp_out)):
                inp_words = sent_inp.split()
                out_words = sent_out.split()
                if setting.endswith("ground_truth"):
                    bracketed_sent = get_ground_truth_brackets(inp_words, out_words)
                elif setting.endswith("supervised"):
                    parse_inp = benepar.InputSentence(words=inp_words)
                    parse_tree = parser.parse(parse_inp)
                    
                    if "full." in setting:
                        bracketed_sent = get_parse_tree_brackets(parse_tree, allowed_all_constituencies=True)
                    elif "NP+VP." in setting:
                        bracketed_sent = get_parse_tree_brackets(parse_tree, allowed_constituencies=['NP', 'VP'])
                    elif "NP.no_nest." in setting:
                        bracketed_sent = get_parse_tree_brackets(parse_tree, allowed_constituencies=['NP'], allowed_nesting=False)
                    elif "NP." in setting:
                        bracketed_sent = get_parse_tree_brackets(parse_tree, allowed_constituencies=['NP'])
                    elif "VP." in setting:
                        bracketed_sent = get_parse_tree_brackets(parse_tree, allowed_constituencies=['VP'])

                out_sent = []
                c_ind = 0 
                for word in bracketed_sent:
                    if word in ['-LBRAC-', '-RBRAC-']:
                        out_sent.append('O')
                    else:
                        out_sent.append(out_words[c_ind])
                        c_ind += 1
                
                out_in.write(" ".join(bracketed_sent) + "\n")
                out_out.write(" ".join(out_sent) + "\n")


def generate_data(args):
    inp_dir_path = Path(args.input_dir)
    out_dir_path = Path(args.output_dir)

    out_dir_path = out_dir_path / f"{inp_dir_path.name}.{args.setting}"
    shutil.copy(inp_dir_path / "intent_label.txt", out_dir_path)
    shutil.copy(inp_dir_path / "slot_label.txt", out_dir_path)
    
    for split in ["train", "test", "dev"]:
        inp_data_dir = inp_dir_path / split
        out_data_dir = out_dir_path / split
        out_data_dir.mkdir(parents=True, exist_ok=True)
        write_data(inp_data_dir, out_data_dir, args.setting)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--input_dir", default=None, required=True, type=str, help="The path to the input directory with train/test/dev datasets")
    arg_parser.add_argument("--output_dir", default=None, required=True, type=str, help="The path to the output directory")
    arg_parser.add_argument("--setting", default=None, required=True, type=str, help="The type of pre-processing to perform on the input dataset")
    args = arg_parser.parse_args()
    generate_data(args)
