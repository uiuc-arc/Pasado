import sys
sys.path.insert(1, '../')
sys.path.insert(1, '../../forward_mode_non_tensorized_src')
from zono_rev import *


def test_log():
    input_zono = RevAffine([8, 10])

    # total_output = z_log(input_zono)
    total_output = precise_z_log(input_zono)

    total_output.backward()

    print(input_zono.grad.interval)


def test_sigmoid():
    input_zono = RevAffine([1, 1.1])

    total_output = z_sigmoid(input_zono)
    # total_output = precise_z_sigmoid(input_zono)

    total_output.backward()

    print(input_zono.grad.interval)


def test_tanh():
    input_zono = RevAffine([1, 1.1])

    total_output = z_tanh(input_zono)
    # total_output = precise_z_tanh(input_zono)

    total_output.backward()

    print(input_zono.grad.interval)


def test_sin():
    input_zono = RevAffine([1, 1.1])

    # total_output = z_sin(input_zono)
    total_output = precise_z_sin(input_zono)

    total_output.backward()

    print(input_zono.grad.interval)


def test_exp():
    input_zono = RevAffine([1.5, 6])

    total_output = z_exp(input_zono)
    # total_output = precise_z_exp(input_zono)

    total_output.backward()

    print(input_zono.grad.interval)


def test_normal_cdf():
    input_zono = RevAffine([1.5, 1.6])

    # total_output = z_normal_cdf(input_zono)
    total_output = precise_z_normal_cdf(input_zono)

    total_output.backward()

    print(input_zono.grad.interval)


def test_sqrt():
    # input_zono = Interval(2,3)
    input_zono = RevAffine([5, 11])

    total_output = z_sqrt(input_zono)
    # total_output = precise_z_sqrt(input_zono)

    total_output.backward()

    print(input_zono.grad.interval)


def test_add():
    input_zono = RevAffine([1, 2])

    second_input_zono = RevAffine([1, 3])

    total_output = z_add(input_zono, second_input_zono)

    total_output.backward()

    print(input_zono.grad.interval)
    print(second_input_zono.grad.interval)


def test_mul():
    input_zono = RevAffine([1, 2])

    second_input_zono = RevAffine([1, 22])

    total_output = z_mul(input_zono, second_input_zono)

    total_output.backward()

    print(input_zono.grad.interval)
    print(second_input_zono.grad.interval)


def test_div():
    input_zono = RevAffine([-1, 2])

    second_input_zono = RevAffine([10, 10.4])

    # total_output = z_div(input_zono, second_input_zono)
    total_output = precise_z_div(input_zono, second_input_zono)

    total_output.backward()

    print(input_zono.grad.interval)
    print(second_input_zono.grad.interval)


if __name__ == '__main__':
    # test_tanh()
    # test_sin()
    # test_normal_cdf()
    # test_sigmoid()
    # test_sqrt()
    # test_log()
    # test_div()
    # test_mul()
    # test_add()
    test_exp()
