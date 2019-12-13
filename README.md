# py2tikz
Utilities to write TeX/TikZ from python

Dependencies
------------
- a working TeX installation with pdflatex
- numpy 
- pyyaml 


Usage
-----
To produce the TeX source code output, call the python script

```python -i <input.yaml> -o <output.tex> [-w <wrapper.tex>]```

The resulting source can be included in a larger TeX project
via `\input{path/to/file}`.
To create a stand-alone PDF of the plot, use the `-w` flag
to specify a second output file.
This optional wrapper file can be compiled directly to
produce a PDF.


 
As a convencience, the script `gen-fig` can run the python
script and compile the stand-alone PDF.
The script take three (required) positional arguments
```
gen-fig /path/to/input.yaml /path/to/output.tex /path/to/wrapper.tex
```
and uses pdflatex to emit a PDF file with the same name as
the wrapper (differing by a file extension).


Make sure to
```
chmod +x gen-fig
```
to use this.


YAML inputs
-----------
The data to plot can be specified as:
- the path to suitably formatted tex file (see `test.yaml` and `test_data.tex`)
- the path to a numpy file (.npz) (see `test_npz.yaml`)
- a string expressing a TikZ-compatible equation (see, e.g., `test_multi.yaml`)


Formatting is specified in the YAML block, including per-data series options.
See the example YAML documents for details.


Known issues
------------
The `test_npz.yaml` refers to a file not included in the repo. 
Feel free to either change the file name and data key references in the YAML
document, or the following would do the trick:
```
import numpy as np

N = 50
OUTFILE = "npz\_to\_plot.npz"

iters = np.linspace(0, 10, N)
residuals = np.random.random(N)
np.savez(OUTFILE, iters=iters, residuals=residuals)
```














