import splitfolders

inputpath = '../../2020Dset/'
outpath = '../../2020Dset/'

splitfolders.ratio(inputpath, output=outpath, seed=1337, ratio=(0.95, 0.05))

# CLI
# Usage:
#     splitfolders [--output] [--ratio] [--fixed] [--seed] [--oversample] [--group_prefix] folder_with_images
# Options:
#     --output        path to the output folder. defaults to `output`. Get created if non-existent.
#     --ratio         the ratio to split. e.g. for train/val/test `.8 .1 .1 --` or for train/val `.8 .2 --`.
#     --fixed         set the absolute number of items per validation/test set. The remaining items constitute
#                     the training set. e.g. for train/val/test `100 100` or for train/val `100`.
#     --seed          set seed value for shuffling the items. defaults to 1337.
#     --oversample    enable oversampling of imbalanced datasets, works only with --fixed.
#     --group_prefix  split files into equally-sized groups based on their prefix
# Example:
#     splitfolders --ratio .8 .1 .1 -- folder_with_images
