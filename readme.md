# Parallel Computing Assignment 1 - Jose Rodriguez

The report for the first part of this assignment is contained inside of
`report-part-1.md`. The report for the second part of the assignment is
contained inside of `report-part-ii.pdf`.

Running `sh run.sh` will run the tests that will run the performance tests
shown in the report.

## Instructions

For full instructions on how to run do:

```sh
$ python3 matrixMultiply --help
```

but the tldr, is that if you want to run it by reading two files from different
matrices do:

```sh
    $ ... --file1 a.txt --file2 b.txt
```

for two matrices of n size do:

```sh
$ ... --size 100
```

to run small functional tests do:

```sh
$ ... --test true
```

output matrix is put inside of output.txt, but can be changed with the
--output flag.

You can generate test input files through the following commands:

```sh
$ python3 matrixMultiply.py --example true
```

After generating the input files, you can run:

```sh
$ python3 matrixMultiply.py --file1 a.txt --file2 b.txt
```
