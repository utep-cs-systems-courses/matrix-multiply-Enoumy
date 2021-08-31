# Parallel Computing Assignment 1 - Jose Rodriguez

For full instructions on how to run do:

```sh
$ python3 serialMatrixMultiply --help 
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
$ python3 serialMatrixMultiply.py --example true
```


After generating the input files, you can run:

```sh
$ python3 serialMatrixMultiply.py --file1 a.txt --file2 b.txt
```