from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NewtonResult:
    solution: np.ndarray
    residual_norms: list[float]
    step_lengths: list[float]
    iterates: list[np.ndarray]
    converged: bool
    iterations: int


def bvp_residual(u: np.ndarray, num_interior_points: int = 200) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    if u.ndim != 1 or u.size != num_interior_points:
        raise ValueError(f"expected a vector of length {num_interior_points}")

    h = 1.0 / (num_interior_points + 1)
    extended = np.zeros(num_interior_points + 2, dtype=float)
    extended[1:-1] = u
    laplace_term = (-extended[:-2] + 2.0 * extended[1:-1] - extended[2:]) / h**2
    return laplace_term + u**3 - 1.0


def bvp_jacobian(u: np.ndarray, num_interior_points: int = 200) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    if u.ndim != 1 or u.size != num_interior_points:
        raise ValueError(f"expected a vector of length {num_interior_points}")

    h = 1.0 / (num_interior_points + 1)
    main_diag = 2.0 / h**2 + 3.0 * u**2
    off_diag = -np.ones(num_interior_points - 1, dtype=float) / h**2
    jacobian = np.diag(main_diag)
    jacobian += np.diag(off_diag, k=-1)
    jacobian += np.diag(off_diag, k=1)
    return jacobian


def solve_bvp(
    num_interior_points: int = 200,
    initial_guess: np.ndarray | None = None,
    tolerance: float = 1e-12,
    max_iterations: int = 30,
    armijo: float = 1e-4,
    min_step_length: float = 1e-10,
) -> NewtonResult:
    if initial_guess is None:
        u = np.zeros(num_interior_points, dtype=float)
    else:
        u = np.asarray(initial_guess, dtype=float).copy()
        if u.ndim != 1 or u.size != num_interior_points:
            raise ValueError(f"expected a vector of length {num_interior_points}")

    residual = bvp_residual(u, num_interior_points)
    residual_norms = [float(np.linalg.norm(residual))]
    step_lengths: list[float] = []
    iterates = [u.copy()]

    for iteration in range(1, max_iterations + 1):
        if residual_norms[-1] <= tolerance:
            return NewtonResult(
                solution=u,
                residual_norms=residual_norms,
                step_lengths=step_lengths,
                iterates=iterates,
                converged=True,
                iterations=iteration - 1,
            )

        jacobian = bvp_jacobian(u, num_interior_points)
        step = np.linalg.solve(jacobian, -residual)
        merit = 0.5 * residual_norms[-1] ** 2

        step_length = 1.0
        while step_length >= min_step_length:
            trial = u + step_length * step
            trial_residual = bvp_residual(trial, num_interior_points)
            trial_merit = 0.5 * float(np.dot(trial_residual, trial_residual))
            if trial_merit <= (1.0 - armijo * step_length) * merit:
                u = trial
                residual = trial_residual
                break
            step_length *= 0.5
        else:
            if residual_norms[-1] <= 10.0 * tolerance:
                return NewtonResult(
                    solution=u,
                    residual_norms=residual_norms,
                    step_lengths=step_lengths,
                    iterates=iterates,
                    converged=True,
                    iterations=iteration - 1,
                )
            return NewtonResult(
                solution=u,
                residual_norms=residual_norms,
                step_lengths=step_lengths,
                iterates=iterates,
                converged=False,
                iterations=iteration - 1,
            )

        residual_norms.append(float(np.linalg.norm(residual)))
        step_lengths.append(step_length)
        iterates.append(u.copy())

    return NewtonResult(
        solution=u,
        residual_norms=residual_norms,
        step_lengths=step_lengths,
        iterates=iterates,
        converged=residual_norms[-1] <= tolerance,
        iterations=max_iterations,
    )
