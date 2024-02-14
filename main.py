
from coastline_trenches_edge_detection import canny_detection
from zero_crossing_edge_detection import zero_crossing
import argparse

parser = argparse.ArgumentParser(
                    prog='Trenches and Crests Edge Detection',
                    description='Can use the Canny method or Zero crossing using both LoG and DoG kernels')
parser.add_argument('-i', '--input_path')
parser.add_argument('-m', '--method', choices=['canny', 'zero-dog','zero-log'])
parser.add_argument('-f', '--output_format', choices=['xyz','shp'])
args = parser.parse_args()
if args['method'] == 'canny':
    canny_detection(args['input_path'], args['output_format'])
elif args['method'] == 'zero-dog':
    zero_crossing(args['input_path'], 'dog',args['output_format'])
elif args['method'] == 'zero-log':
    zero_crossing(args['input_path'], 'log', args['output_format'])


