"""
Parallel Computing Assignment 1 - Jose Rodriguez

For full instructions on how to run do:

    ```
    $ python3 serialMatrixMultiply --help 
    ```

but the tldr, is that if you want to run it by reading two files from different
matrices do:

    $ ... --file1 a.txt --file2 b.txt

for two matrices of n size do:
    $ ... --size 100


to run small functional tests do:
    $ ... --test true


output matrix is put inside of output.txt, but can be changed with the
--output flag.
"""


import argparse
import matrixUtils
import time


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


def identityMatrixTest():
    a = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    b = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    out = [[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]]

    assert matrixMultiply(a, b) == out, 'identity matrix test'
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

    assert matrixMultiply(a, b) == out, 'square matrix test'
    print('square matrix test passed!')


def differentShapesTest():
    a = [[1, 2, 3],
         [4, 5, 6]]
    b = [[7, 8],
         [9, 10],
         [11, 12]]
    out = [[58, 64],
           [139, 154]]

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

    args = parser.parse_args()

    if args.test:
        [t() for t in tests]

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

    start = time.clock_getttime(time.CLOCK_MONOTONIC)
    resulting_matrix = matrixMultiply(matrix1, matrix2)
    time_taken = time.clock_getttime(time.CLOCK_MONOTONIC) - start


print('A x B (cropped):')
print(f'Time taken: {time_taken} seconds')
matrixUtils.printSubarray(resulting_matrix, size=args.show_size)

matrixUtils.writeToFile(resulting_matrix, args.output)


if __name__ == '__main__':
    main()
