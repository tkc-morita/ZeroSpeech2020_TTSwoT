# coding: utf-8

import pandas as pd
import scipy.io.wavfile as spw
import os, argparse, glob

# def create_data_director(path, root, ext='.wav'):
# 	rows = []
# 	with open(path, 'r') as f:
# 		for line in f.readlines():
# 			line = line.rstrip()
# 			filename_wo_ext,onset,offset = line.split(' ')
# 			speaker = filename_wo_ext.split('_')[0]
# 			filename = filename_wo_ext+ext
# 			data_type = get_data_type(filename, root)
# 			rows.append([filename, speaker, onset, offset, data_type])
# 	return pd.DataFrame(rows, columns=['filename','speaker','onset','offset','data_type'])

def create_data_director(source_dirs):
	root = find_root(source_dirs)
	rows = []
	for d in source_dirs:
		for filepath in glob.glob(os.path.join(d, '*.wav')):
			filename = os.path.basename(filepath)
			speaker = filename.split('_')[0]
			rel_path = os.path.relpath(filepath, root)
			fs,wav = spw.read(filepath)
			duration = wav.shape[0] / float(fs)
			rows.append([rel_path, speaker, duration, os.path.splitext(filename)[0]])
	return pd.DataFrame(rows, columns=['filename','speaker','duration','filebase'])

def find_root(dirs):
	root = ''
	for dirnames in zip(*[d.split('/') for d in dirs]):
		if len(set(dirnames))==1:
			root += dirnames[0]+'/'
		else:
			break
	return root

# def get_data_type(filename, root):
# 	paths = glob.glob(os.path.join(root, '**', filename), recursive=True)
# 	num_files = len(paths)
# 	assert num_files<=1, 'Multiple copies of {filename}, at {paths}'.format(filename=filename, paths=paths)
# 	if num_files==0:
# 		print('Missing file: {}'.format(filename))
# 		return '__missing__'
# 	path = paths[0]
# 	data_type = os.path.relpath(os.path.dirname(path), start=root).strip('/').replace('/','_')
# 	return data_type

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('source_dirs', type=str, nargs='+', help='Path to the directories containing wav files.')
	parser.add_argument('save_path', type=str, help='Path to the csv where processed data are saved.')
	parser.add_argument('--synthesis', type=str, default=None, help='Path to the txt file listing source wavs and speakers to be synthesized.')
	args = parser.parse_args()

	df = create_data_director(args.source_dirs)

	if not args.synthesis is None:
		df_syn = pd.read_csv(args.synthesis, names=['filebase','synthesis_speaker'], sep=' ')
		df_syn.loc[:,'filebase'] = df_syn.filebase.map(os.path.basename)
		df = df.merge(df_syn, how='left', on='filebase')
	df = df.drop(columns=['filebase'])

	save_dir = os.path.dirname(args.save_path)
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	df.to_csv(args.save_path, index=False)
