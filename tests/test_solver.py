import numpy as np

from nonlinear_bvp import bvp_jacobian, bvp_residual, solve_bvp


def test_residual_and_jacobian_match_problem_size():
    n = 200
    u = np.linspace(0.0, 0.2, n)

    residual = bvp_residual(u, n)
    jacobian = bvp_jacobian(u, n)

    assert residual.shape == (n,)
    assert jacobian.shape == (n, n)
    assert np.allclose(np.diag(jacobian, k=1), -(n + 1) ** 2)
    assert np.allclose(np.diag(jacobian, k=-1), -(n + 1) ** 2)


def test_newton_solver_converges_for_n_200():
    result = solve_bvp(num_interior_points=200, tolerance=1e-11, max_iterations=20)

    assert result.converged
    assert result.residual_norms[-1] <= 1e-11
    assert np.allclose(result.solution, result.solution[::-1], atol=1e-12)
    assert all(step > 0.0 for step in result.step_lengths)


def test_newton_is_quadratic_near_solution():
    n = 200
    reference = solve_bvp(num_interior_points=n, tolerance=1e-11, max_iterations=20)
    assert reference.converged

    mode = np.sin(np.pi * np.arange(1, n + 1) / (n + 1))
    mode /= np.linalg.norm(mode)
    initial_guess = reference.solution + 3e-2 * mode

    result = solve_bvp(
        num_interior_points=n,
        initial_guess=initial_guess,
        tolerance=1e-11,
        max_iterations=6,
    )

    assert result.converged
    assert result.step_lengths == [1.0, 1.0]

    errors = [np.linalg.norm(iterate - reference.solution) for iterate in result.iterates]
    quadratic_ratios = [
        errors[k + 1] / errors[k] ** 2
        for k in range(len(errors) - 1)
        if errors[k] > 1e-14 and errors[k + 1] > 0.0
    ]

    assert len(quadratic_ratios) >= 2
    assert max(quadratic_ratios[:2]) < 1e-2
