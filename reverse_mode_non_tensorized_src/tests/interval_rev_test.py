import sys
sys.path.insert(1, '../')
sys.path.insert(1, '../../forward_mode_non_tensorized_src')
from interval_rev import *


def test_log():
    input_zono = RevInterval([8, 10])

    total_output = i_log(input_zono)
    print(total_output)

    total_output.backward()

    print(input_zono.grad)


def test_sigmoid():
    input_zono = RevInterval([1, 1.1])

    total_output = i_sigmoid(input_zono)
    print(total_output)

    total_output.backward()

    print(input_zono.grad)


def test_tanh():
    input_zono = RevInterval([1, 1.1])

    total_output = i_tanh(input_zono)
    print(total_output)

    total_output.backward()

    print(input_zono.grad)


def test_sin():
    input_zono = RevInterval([1, 1.1])

    total_output = i_sin(input_zono)
    print(total_output)

    total_output.backward()

    print(input_zono.grad)


def test_exp():
    input_zono = RevInterval([1.5, 6])

    total_output = i_exp(input_zono)
    print(total_output)

    total_output.backward()

    print(input_zono.grad)


def test_normal_cdf():
    input_zono = RevInterval([1.5, 1.6])

    total_output = i_normal_cdf(input_zono)
    print(total_output)

    total_output.backward()

    print(input_zono.grad)


def test_sqrt():
    input_zono = RevInterval([5, 11])

    total_output = i_sqrt(input_zono)
    print(total_output)

    total_output.backward()

    print(input_zono.grad)


def test_add():
    input_zono = RevInterval([1, 2])

    second_input_zono = RevInterval([1, 3])

    total_output = i_add(input_zono, second_input_zono)
    print(total_output)

    total_output.backward()

    print(input_zono.grad)
    print(second_input_zono.grad)


def test_mul():
    input_zono = RevInterval([1., 2.])

    second_input_zono = RevInterval([1., 22.])

    total_output = i_mul(input_zono, second_input_zono)
    print(total_output)

    total_output.backward()

    print(input_zono.grad)
    print(second_input_zono.grad)


def test_div():
    input_zono = RevInterval([-1, 2])

    second_input_zono = RevInterval([8, 10.4])

    total_output = i_div(input_zono, second_input_zono)
    print(total_output)

    total_output.backward()

    print(input_zono.grad)
    print(second_input_zono.grad)


if __name__ == '__main__':
    test_tanh()
    # test_sin()
    # test_normal_cdf()
    # test_sigmoid()
    # test_sqrt()
    # test_log()
    # test_div()
    # test_mul()
    # test_add()
    # test_exp()
