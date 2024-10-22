# Pre-Postprocessing
Interactive processing of image files from a specified folder 
with demonstration of results in (matplotlib) plots format.

Includes:

1) preprocessing - preparing the image for box detection,
2) postprocessing - calculating the boxes on the image with results validation.

Running:

The program runs in the terminal in interactive mode, examples of run:

python3 -i *.py -J./data/			            - processing files from a folder './data/'
python3 -i *.py -J./data/ -s1800		      - processing files from folder './data/' with resize
python3 -i *.py -J./data/cm/ -s1800 -i		- processing files from folder './data/cm/' with resize and inversion of pixel intensities
python3 -i *.py -J./data/cm/ -i -n76 		   - processing files from folder './data/cm/', starting from n-flie (#76) and with inversion of pixel intensities

Usage interative mode (control keys description):

After preprocessing the current file, it shows the timing of the basic operations 
and waits for a control key to be pressed to continue:

- Enter 	- processing the next file in the folder;
- 'q'+Enter - exit to terminal;
- 'a'+Enter - run advanced postprocessing (building boxes, binding boxes to lines, validating results);
- 'b'+Enter - run simple postprocessing (only building boxes);
- 's'+Enter - enabling streaming (without waiting for the control command), only preprocessing.

Libraries used:

numpy		    - for all matrix operations
opencv		  - for reading files and special functions
matplotlib	- for plot results
scipy		    - for image resize

