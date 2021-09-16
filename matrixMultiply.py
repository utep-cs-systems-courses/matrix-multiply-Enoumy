"""
# Parallel Computing Assignment 1 - Jose Rodriguez

For full instructions on how to run do:

```sh
$ python3 matrixMultiply.py --help
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
"""


import argparse
import matrixUtils
import time
import pymp


def matrixMultiply(matrix_a, matrix_b):
    """
    Returns a the matrix multiplication of two matrices.
    The first matrix is the left matrix and the second parameter matrix is the
    right matrix.
    """
    assert len(matrix_a[0]) == len(matrix_b), 'Both matrices can be multiplied'

    out = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]

    for row in range(len(out)):
        for col in range(len(out[0])):
            out[row][col] = sum(matrix_a[row][i] * matrix_b[i][col]
                                for i in range(len(matrix_b)))

    return out


def blockedMatrixMultiply(matrix_a, matrix_b):
    """
    Returns a the matrix multiplication of two matrices.
    The first matrix is the left matrix and the second parameter matrix is the
    right matrix.
    """
    assert len(matrix_a[0]) == len(matrix_b), 'Both matrices can be multiplied'

    out = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]

    tile_size = 16

    for row in range(0, len(out), tile_size):
        for col in range(0, len(out[0]), tile_size):
            for i in range(len(matrix_b)):
                j_end_val = col + tile_size
                for j in range(col, min(j_end_val, len(matrix_a[0]))):
                    k_end_val = row + tile_size
                    curr_sum = out[i][j]
                    for k in range(row, min(k_end_val, len(matrix_b))):
                        curr_sum += matrix_a[i][k] * matrix_b[k][j]
                    out[i][j] = curr_sum

    return out


def matrixMultiplyParallelRow(matrix_a, matrix_b):
    """
    Returns a the matrix multiplication of two matrices.
    The first matrix is the left matrix and the second parameter matrix is the
    right matrix.
    """
    assert len(matrix_a[0]) == len(matrix_b), 'Both matrices can be multiplied'

    out = pymp.shared.array((len(matrix_a), len(matrix_b[0])), dtype="uint32")

    with pymp.Parallel() as p:
        print(f'Number of thread: {p.thread_num} of {p.num_threads}')

        for row in p.range(0, len(out)):
            for col in range(len(out[0])):
                out[row][col] = sum(matrix_a[row][i] * matrix_b[i][col]
                                    for i in range(len(matrix_b)))

    return list(list(l) for l in out)


matrix_functions = [matrixMultiply,
                    blockedMatrixMultiply, matrixMultiplyParallelRow]


def identityMatrixTest():
    a = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    b = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    out = [[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]]

    for f in matrix_functions:
        assert f(a, b) == out, 'identity matrix test'
    print('identity test passed!')


def squareMatricTest():
    a = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    b = [[10, 11, 12],
         [13, 14, 15],
         [16, 17, 18]]
    out = [[84, 90, 96],
           [201, 216, 231],
           [318, 342, 366]]

    for f in matrix_functions:
        assert f(a, b) == out, 'square matrix test'
    print('square matrix test passed!')


def differentShapesTest():
    a = [[1, 2, 3],
         [4, 5, 6]]
    b = [[7, 8],
         [9, 10],
         [11, 12]]
    out = [[58, 64],
           [139, 154]]

    for f in matrix_functions:
        assert matrixMultiply(a, b) == out, 'different shapes test'
    print('different shapes test passsed!')


tests = [identityMatrixTest, squareMatricTest, differentShapesTest]


def main():
    parser = argparse.ArgumentParser(
        description='Multiplies two matrices!'
    )

    parser.add_argument('--file1', default=None, type=str,
                        help="file in which the first matrix is found.")
    parser.add_argument('--file2', default=None, type=str,
                        help="file in which the second matrix is found.")
    parser.add_argument('--size', default=10, type=int,
                        help="size to use in order to generate temporary matrices if no files are given")
    parser.add_argument('--output', default="output.txt",
                        type=str, help='file to put the output of the matrix multiplication')
    parser.add_argument('--test', default=False, type=bool,
                        help='If present, runs tests')
    parser.add_argument('--show_size', default=10, type=int,
                        help='Maximum matrix size to print (rest is ignored) (Whole matrix is saved to file).')
    parser.add_argument('--example', default=False, type=bool,
                        help='If set to true, program will only generate test data into a.txt and b.txt')

    args = parser.parse_args()

    if args.test:
        [t() for t in tests]
        return

    if args.example:
        matrix = matrixUtils.genMatrix(size=500)
        matrixUtils.writeToFile(matrix, 'a.txt')
        matrixUtils.writeToFile(matrix, 'b.txt')
        return

    if args.file1 and args.file2:
        matrix1 = matrixUtils.readFromFile(args.file1)
        matrix2 = matrixUtils.readFromFile(args.file2)
    else:
        matrix1 = matrixUtils.genMatrix(size=args.size)
        matrix2 = matrixUtils.genMatrix(size=args.size)

    print('Matrix A (cropped):')
    matrixUtils.printSubarray(matrix1, size=args.show_size)

    print('Matrix B (cropped):')
    matrixUtils.printSubarray(matrix2, size=args.show_size)

    start = time.clock_gettime(time.CLOCK_MONOTONIC)
    resulting_matrix = matrixMultiply(matrix1, matrix2)
    time_taken = time.clock_gettime(time.CLOCK_MONOTONIC) - start

    print('A x B (cropped):')
    print(f'Time taken: {time_taken} seconds')
    matrixUtils.printSubarray(resulting_matrix, size=args.show_size)

    start = time.clock_gettime(time.CLOCK_MONOTONIC)
    resulting_blocked_matrix = blockedMatrixMultiply(matrix1, matrix2)
    time_taken = time.clock_gettime(time.CLOCK_MONOTONIC) - start

    print('A x B through blocked matrix multiply (cropped):')
    print(f'Time taken: {time_taken} seconds')
    matrixUtils.printSubarray(resulting_blocked_matrix, size=args.show_size)

    start = time.clock_gettime(time.CLOCK_MONOTONIC)
    resulting_parallel_matrix = matrixMultiplyParallelRow(matrix1, matrix2)
    time_taken = time.clock_gettime(time.CLOCK_MONOTONIC) - start

    print('A x B through row parallel matrix multiply (cropped):')
    print(f'Time taken: {time_taken} seconds')
    matrixUtils.printSubarray(resulting_parallel_matrix, size=args.show_size)

    matrixUtils.writeToFile(resulting_matrix, args.output)


if __name__ == '__main__':
    main()
