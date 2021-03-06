# Assignment 1 - Matrix multiplication that's serial

During this assignment I implemented a serial matrix multiplication
algorithm. I used python in the assingment. The program is able to
multiply matrices like in a matrix algebra course. The program can either
multiply square 1's matrices or load matrices from files.

For full instructions on how to run do:

```sh
$ python3 serialMatrixMultiply.py --help
```

but the tldr, is that if you want to run it by reading two files from different
matrices do:

```sh
$ python3 serialMatrixMultiply.py --file1 a.txt --file2 b.txt
```

For two matrices of n size do:

```sh
$ python3 serialMatrixMultiply.py --size 100
```

To run small functional tests do:

```sh
$ python3 serialMatrixMultiply.py --test true
```

Output matrix is put inside of output.txt, but can be changed with the
`--output` flag.

You can generate test input files through the following commands:

```sh
$ python3 serialMatrixMultiply.py --example true
```

After generating the input files, you can run:

```sh
$ python3 serialMatrixMultiply.py --file1 a.txt --file2 b.txt
```

Additionally, you can run the program that took ~10 seconds on my machine with
a matrix of size 400 by doing:

```sh
$ python3 serialMatrixMultiply.py --size 400

or

$ source run.sh
```

## Blocked matrix multiply extension.

The first extension to the matrix multiply algorithm was adding a blocked
version of the algorithm in which there are smaller subsections for each
matrix. There are now "blocks". The behavior ended up being the same in regards
to the output outputted by the program, but after timing both versions,
on a size of 400, the blocked version took around twice as much.

This makes sense due to the extra overhead of the extra loops, and the irregular
array accesses.
