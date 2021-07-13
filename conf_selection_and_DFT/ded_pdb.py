#!/usr/bin/env python
import numpy as np
import sys as sys
import argparse as ap
import pathlib as pl

class pdb:
	def __init__(self, xyz, data, head=None, col=None):
		self.xyz = xyz # pl.Path object
		self.data = data # pl.Path object
		self.path = self.xyz.parents[0]
		self.name = self.data.stem
		
		if head == None:
			self.head = 0
		else:
			self.head = head
		
		if col == None:
			self.col = 0
		else:
			self.col = col
		
		self.read(self.head, self.col)
	
	def read(self, head, col):
		self.xyz = np.genfromtxt(self.xyz, delimiter=None, usecols = (0, 1, 2), skip_header = head)
		self.values = np.genfromtxt(self.data, delimiter=None, usecols = (col), skip_header = 0)
		self.npoints = len(self.xyz)
		return
		
	def dump(self, name):
		with open(self.path / pl.Path(name + ".pdb") , 'w') as f:
			f.write('REMARK   ' + str(self.npoints) + '\n')
			for ni in range(self.npoints):
				f.write('HETATM')
				f.write('{:>{length}s}'.format(str(ni+1), length = 5))
				f.write('{:>{length}s}'.format('C', length = 3))
				f.write('{:>{length}s}'.format('MOL', length = 6))
				f.write('{:>{length}s}'.format('A', length = 2))
				f.write('{:>{length}s}'.format('1', length = 4))
				f.write('{:>{length}s}'.format('', length = 4))
				f.write('{:>{length}s}'.format(str(format(self.xyz[ni][0], '.3f')), length = 8))
				f.write('{:>{length}s}'.format(str(format(self.xyz[ni][1], '.3f')), length = 8))
				f.write('{:>{length}s}'.format(str(format(self.xyz[ni][2], '.3f')), length = 8))
				f.write('{:>{length}s}'.format('1.00', length = 6))
				f.write('{:>{length}s}'.format(str(format(self.values[ni], '.2f')), length = 6))
				f.write('{:>{length}s}'.format('C', length = 11))
				f.write('\n')
		return

def main():
	# Parse arguments
	parser = ap.ArgumentParser()
	parser.add_argument("points", help="List of points (x y z)", type=str)
	parser.add_argument("values", help="List of values at (x y z) positions", type=str)
	parser.add_argument("-c", "--col", help="Number of column of values in values file.", type=int, default=1)
	parser.add_argument("-e", "--head", help="Number of header lines in points file.", type=int, default=0)
	args = parser.parse_args()
	
	fxyz = pl.Path(sys.argv[1]).resolve()
	fdata = pl.Path(sys.argv[2]).resolve()
	fname = fdata.stem
	data = pdb(fxyz, fdata, args.head, args.col-1)
	data.dump(str(fname))


if __name__ == "__main__":
	main()
