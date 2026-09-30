import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns


def get_minor(A, i, j):
    matrix = A.tolist()
    minor = np.array([row[:j] + row[(j + 1):]]
                     for row in (matrix[:i] + matrix[(i + 1):]))
    return minor


def determinant(A):
    n = len(A)
    if n == 1:
        return A[0, 0]
    if n == 2:
        return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]

    det = 0
    for j in range(n):
        det += ((-1) ** j) * A[0, j] * determinant(get_minor(A, 0, j))
    return det


def gauss_solve(A, b):
    x = np.zeros(A.shape[1], dtype="float")
    A_extended = np.hstack([A, b.reshape(-1, 1)])
    rows, cols = A_extended.shape

    for i in range(rows):
        coef = A_extended[i, i]
        A_extended[i, :] /= coef
        for k in range(i + 1, rows):
            row_coef = A_extended[k, i]
            A_extended[k, :] -= row_coef * A_extended[i, :]

    A_new = A_extended[:, :A.shape[1]]
    b_new = A_extended[:, -1]

    for i in range(rows - 1, -1, -1):
        x[i] = b_new[i] - A_new[i, (i+1):] @ x[(i+1):]

    return x


def solve_traversal(A, b, c):
    n_variables = A.shape[1]
    n_restrictions = A.shape[0]
    eps = 1e-9

    vertices = []

    for indices_comb in combinations(range(n_restrictions), n_variables):
        A_subarray = np.array([A[i] for i in indices_comb])
        b_subarray = np.array([b[i] for i in indices_comb])

        if abs(determinant(A_subarray)) < eps:
            continue

        try:
            x = gauss_solve(A_subarray, b_subarray)
        except:
            continue

        if all(A[i] @ x - b[i] <= eps for i in range(n_restrictions)):
            vertices.append(x)

    sorted_vertices = sorted(vertices, key=lambda v: v @ c)[::-1]
    costs = [v @ c for v in sorted_vertices]

    return sorted_vertices, costs


def visualize(A, b, c, points, zs):
    fig, ax = plt.subplots(figsize=(8, 6))

    lines = [(A[i], b[i]) for i in range(len(b))]
    lines += [(np.array([1.0, 0.0]), 0.0), (np.array([0.0, 1.0]), 0.0)]

    verts = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            M = np.array([lines[i][0], lines[j][0]])
            if abs(np.linalg.det(M)) < 1e-9:
                continue
            x = np.linalg.solve(M, [lines[i][1], lines[j][1]])
            if np.all(A @ x <= b + 1e-9) and np.all(x >= -1e-9):
                if not any(np.linalg.norm(x - v) < 1e-7 for v in verts):
                    verts.append(x)
    verts = np.array(verts)
    center = verts.mean(axis=0)
    verts = verts[np.argsort(np.arctan2(verts[:, 1] - center[1],
                                        verts[:, 0] - center[0]))]

    ax.add_patch(plt.Polygon(verts, closed=True, alpha=0.25,
                             color='lightgreen', label='Допустимая область'))

    x_max = verts[:, 0].max() + 1
    x_vals = np.linspace(0, x_max, 400)
    for i in range(len(b)):
        if abs(A[i, 1]) > 1e-9:
            ax.plot(x_vals, (b[i] - A[i, 0] * x_vals) / A[i, 1],
                    '--', linewidth=1,
                    label=f'{A[i, 0]:g}·x1 + {A[i, 1]:g}·x2 = {b[i]:g}')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    for i, (p, z) in enumerate(zip(points, zs)):
        x, y = float(p[0]), float(p[1])
        ax.plot(x, y, 'ko', markersize=8)
        ax.annotate(f'Итерация {i} (z={z:.2f})\n({x:.2f}, {y:.2f})',
                    (x, y), textcoords="offset points", xytext=(8, 8),
                    fontsize=9)

    x_opt, y_opt = float(points[-1][0]), float(points[-1][1])
    ax.plot(x_opt, y_opt, 'o', markersize=14, label='Оптимум', color='red')

    ax.set_xlim(-0.5, x_max)
    ax.set_ylim(-0.5, verts[:, 1].max() + 1)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Симплекс-метод')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.show()


def solve_simplex(A, b, c, strict_inequality, plot=True):
    eps = 1e-9

    additional_variables = np.where(strict_inequality == 1)[0].shape[0]
    additional_matrix = np.zeros((A.shape[0], additional_variables))
    additional_matrix[np.where(strict_inequality == 1)[
        0], np.arange(0, additional_variables)] = 1

    A_extended = np.hstack([A, additional_matrix])
    c_extended = np.array(
        [el for el in c] + [0 for _ in range(additional_variables)])
    n = A_extended.shape[1]
    n_orig = len(c)

    basis = np.array([c.shape[0] + i for i in range(additional_variables)])
    free = np.array([i for i in range(n) if i not in basis])

    iteration = 0
    solution = np.array([0 for _ in range(len(c_extended))], dtype="float")
    solution[basis] = b
    z = np.sum(solution * c_extended)

    points = [solution[:n_orig].copy()]
    zs = [z]

    print(f"Итерация {iteration}")
    print(f"Решение: {solution}")
    print(f"Целевая функция: {z}", end="\n\n")

    while True:
        A_basis = A_extended[:, basis]
        A_basis_inv = np.linalg.inv(A_basis)
        c_basis = c_extended[basis]

        deltas = []
        for free_idx in free:
            c_free = c_extended[free_idx]
            a_vector_free = A_extended[:, free_idx]
            delta = c_free - c_basis @ A_basis_inv @ a_vector_free
            deltas.append(delta)

        if all(delta < eps for delta in deltas):
            print()
            print(f"Найдено оптимальное решение!")
            print(f"Решение: {solution[:len(c)]}")
            print(f"Целевая функция: {z}")

            if plot:
                visualize(A, b, c, points, zs)

            break
        else:
            free_to_basis_idx = np.argmax(deltas)
            free_to_basis_value = free[free_to_basis_idx]

            a_vector_free = A_extended[:, free_to_basis_value]
            b_hat = A_basis_inv @ b
            a_vector_hat = A_basis_inv @ a_vector_free
            valid_basis_indices = np.where(a_vector_hat > eps)[0]
            if len(valid_basis_indices) == 0:
                print(f"Задача не ограничена")
                break

            thetas = b_hat[valid_basis_indices] / \
                a_vector_hat[valid_basis_indices]
            theta_opt = min(thetas)
            theta_opt_idx = np.argmin(thetas)
            basis_to_free_idx = valid_basis_indices[theta_opt_idx]
            basis_to_free_value = basis[basis_to_free_idx]

            solution_basis = b_hat - a_vector_hat * theta_opt
            solution[basis] = solution_basis
            solution[free_to_basis_value] = theta_opt
            z = np.sum(solution * c_extended)

            points.append(solution[:n_orig].copy())
            zs.append(z.copy())

            free[free_to_basis_idx] = basis_to_free_value
            basis[basis_to_free_idx] = free_to_basis_value

            iteration += 1
            print(f"Итерация {iteration}")
            print(f"Решение: {solution}")
            print(f"Целевая функция: {z}", end="\n\n")
