import numpy as np
import pytest

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_1d():
    x = ti.field(ti.f32, shape=(16))

    @ti.kernel
    def func():
        for i in ti.ndrange((4, 10)):
            x[i] = i

    func()

    for i in range(16):
        if 4 <= i < 10:
            assert x[i] == i
        else:
            assert x[i] == 0


@test_utils.test()
def test_2d():
    x = ti.field(ti.f32, shape=(16, 32))

    t = 8

    @ti.kernel
    def func():
        for i, j in ti.ndrange((4, 10), (3, t)):
            val = i + j * 10
            x[i, j] = val

    func()
    for i in range(16):
        for j in range(32):
            if 4 <= i < 10 and 3 <= j < 8:
                assert x[i, j] == i + j * 10
            else:
                assert x[i, j] == 0


@test_utils.test()
def test_3d():
    x = ti.field(ti.f32, shape=(16, 32, 64))

    @ti.kernel
    def func():
        for i, j, k in ti.ndrange((4, 10), (3, 8), 17):
            x[i, j, k] = i + j * 10 + k * 100

    func()
    for i in range(16):
        for j in range(32):
            for k in range(64):
                if 4 <= i < 10 and 3 <= j < 8 and k < 17:
                    assert x[i, j, k] == i + j * 10 + k * 100
                else:
                    assert x[i, j, k] == 0


@test_utils.test()
def test_tensor_based_3d():
    x = ti.field(ti.i32, shape=(6, 6, 6))
    y = ti.field(ti.i32, shape=(6, 6, 6))

    @ti.kernel
    def func():
        lower = ti.Vector([0, 1, 2])
        upper = ti.Vector([3, 4, 5])
        for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
            x[I] = I[0] + I[1] + I[2]
        for i in range(0, 3):
            for j in range(1, 4):
                for k in range(2, 5):
                    y[i, j, k] = i + j + k

    func()

    for i in range(6):
        for j in range(6):
            for k in range(6):
                assert x[i, j, k] == y[i, j, k]


@test_utils.test()
def test_static_grouped():
    x = ti.field(ti.f32, shape=(16, 32, 64))

    @ti.kernel
    def func():
        for I in ti.static(ti.grouped(ti.ndrange((4, 5), (3, 5), 5))):
            x[I] = I[0] + I[1] * 10 + I[2] * 100

    func()
    for i in range(16):
        for j in range(32):
            for k in range(64):
                if 4 <= i < 5 and 3 <= j < 5 and k < 5:
                    assert x[i, j, k] == i + j * 10 + k * 100
                else:
                    assert x[i, j, k] == 0


@test_utils.test()
def test_static_grouped_static():
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=(16, 4))

    @ti.kernel
    def func():
        for i, j in ti.ndrange(16, 4):
            for I in ti.static(ti.grouped(ti.ndrange(2, 3))):
                x[i, j][I] = I[0] + I[1] * 10 + i + j * 4

    func()
    for i in range(16):
        for j in range(4):
            for k in range(2):
                for l in range(3):
                    assert x[i, j][k, l] == k + l * 10 + i + j * 4


@test_utils.test()
def test_field_init_eye():
    # https://github.com/taichi-dev/taichi/issues/1824

    n = 32

    A = ti.field(ti.f32, (n, n))

    @ti.kernel
    def init():
        for i, j in ti.ndrange(n, n):
            if i == j:
                A[i, j] = 1

    init()
    assert np.allclose(A.to_numpy(), np.eye(n, dtype=np.float32))


@test_utils.test()
def test_ndrange_index_floordiv():
    # https://github.com/taichi-dev/taichi/issues/1829

    n = 10

    A = ti.field(ti.f32, (n, n))

    @ti.kernel
    def init():
        for i, j in ti.ndrange(n, n):
            if i // 2 == 0:
                A[i, j] = i

    init()
    for i in range(n):
        for j in range(n):
            if i // 2 == 0:
                assert A[i, j] == i
            else:
                assert A[i, j] == 0


@test_utils.test()
def test_nested_ndrange():
    # https://github.com/taichi-dev/taichi/issues/1829

    n = 2

    A = ti.field(ti.i32, (n, n, n, n))

    @ti.kernel
    def init():
        for i, j in ti.ndrange(n, n):
            for k, l in ti.ndrange(n, n):
                r = i * n**3 + j * n**2 + k * n + l
                A[i, j, k, l] = r

    init()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    r = i * n**3 + j * n**2 + k * n + l
                    assert A[i, j, k, l] == r


@test_utils.test(ti.cpu)
def test_ndrange_ast_transform():
    n, u, v = 4, 3, 2

    a = ti.field(ti.i32, ())
    b = ti.field(ti.i32, ())
    A = ti.field(ti.i32, (n, n))

    @ti.kernel
    def func():
        # `__getitem__ cannot be called from Python-scope` will be raised if
        # `a[None]` is not transformed to `ti.subscript(a, None)` in ti.ndrange:
        for i, j in ti.ndrange(a[None], b[None]):
            r = i * n + j + 1
            A[i, j] = r

    a[None] = u
    b[None] = v

    func()

    for i in range(n):
        for j in range(n):
            if i < u and j < v:
                r = i * n + j + 1
            else:
                r = 0
            assert A[i, j] == r


@test_utils.test()
def test_grouped_ndrange_star():
    @ti.kernel
    def foo() -> ti.i32:
        ret = 0
        for I in ti.grouped(ti.ndrange(*[[1, 3]] * 3)):
            ret += I[0] + I[1] + I[2]
        return ret

    assert foo() == 36


@test_utils.test()
def test_ndrange_three_arguments():
    @ti.kernel
    def foo():
        for i in ti.ndrange((1, 2, 3)):
            pass

    with pytest.raises(
        ti.TaichiSyntaxError,
        match=r"Every argument of ndrange should be a scalar or a tuple/list like \(begin, end\)",
    ):
        foo()


@test_utils.test()
def test_ndrange_start_greater_than_end():
    @ti.kernel
    def ndrange_test(i1: ti.i32, i2: ti.i32, j1: ti.i32, j2: ti.i32) -> ti.i32:
        n: ti.i32 = 0
        for i, j in ti.ndrange((i1, i2), (j1, j2)):
            n += 1
        return n

    assert ndrange_test(0, 10, 0, 20) == 200
    assert ndrange_test(0, 10, 20, 0) == 0
    assert ndrange_test(10, 0, 0, 20) == 0
    assert ndrange_test(10, 0, 20, 0) == 0


@test_utils.test()
def test_ndrange_non_integer_arguments():
    @ti.kernel
    def example():
        for i in ti.ndrange((1.1, 10.5)):
            pass

    with pytest.raises(
        ti.TaichiTypeError,
        match=r"Every argument of ndrange should be an integer scalar or a tuple/list of \(int, int\)",
    ):
        example()


@test_utils.test()
def test_ndrange_should_accept_numpy_integer():
    a, b = np.int64(0), np.int32(10)

    @ti.kernel
    def example():
        for i in ti.ndrange((a, b)):
            pass

    example()


@test_utils.test()
def test_static_ndrange_non_integer_arguments():
    @ti.kernel
    def example():
        for i in ti.static(ti.ndrange(0.1, 0.2, 0.3)):
            pass

    with pytest.raises(
        ti.TaichiTypeError,
        match=r"Every argument of ndrange should be an integer scalar or a tuple/list of \(int, int\)",
    ):
        example()


@test_utils.test()
def test_static_ndrange_should_accept_numpy_integer():
    a, b = np.int64(0), np.int32(10)

    @ti.kernel
    def example():
        for i in ti.static(ti.ndrange((a, b))):
            pass

    example()


@test_utils.test(exclude=[ti.amdgpu])
def test_n_loop_var_neq_dimension():
    @ti.kernel
    def iter():
        for i in ti.ndrange(1, 4):
            print(i)

    with pytest.warns(
        DeprecationWarning,
        match="Ndrange for loop with number of the loop variables not equal to",
    ):
        iter()


@test_utils.test()
def test_2d_loop_over_ndarray():
    @ti.kernel
    def foo(arr: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        M = arr.shape[0]
        for i, j in ti.ndrange(M, M):
            verts = ti.math.vec4(arr[i], arr[i + 1], arr[j], arr[j + 1])

    array = ti.ndarray(ti.i32, shape=(16,))
    foo(array)
