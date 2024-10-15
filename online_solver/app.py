
from flask import Flask, request, render_template, redirect, url_for, session, flash
from sympy import Matrix, symbols, linsolve, eye
import numpy as np
from scipy.linalg import lu
import json
import os
import logging

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)

# Путь к файлу для хранения истории
history_file = 'history.json'

# Загрузка истории из файла
def load_history():
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    return []

# Сохранение истории в файл
def save_history(history):
    with open(history_file, 'w') as f:
        json.dump(history, f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        system = request.form.getlist('equations')
        method = request.form.get('method')  # Получаем выбранный метод решения

        if not method:
            logging.error("Method is not provided in form!")
            flash("Ошибка: метод не выбран!")
            return render_template('index.html', history=load_history())

        logging.debug(f"Method selected from form: {method}")

        history = load_history()
        history.append({'system': system, 'method': method})
        save_history(history)

        return redirect(url_for('solve', method=method))
    
    return render_template('index.html', history=load_history())

@app.route('/solve', methods=['POST', 'GET'])
def solve():
    method = request.args.get('method')  # Получаем метод через аргументы запроса

    if not method:
        logging.error("Method is not provided in request args!")
        return "Ошибка: метод не передан!"
    
    logging.debug(f"Solving method received: {method}")

    equations = request.form.getlist('equations')
    if not equations:
        logging.error("Equations are missing!")
        return "Ошибка: уравнения не получены!"
    
    logging.debug(f"Equations received: {equations}")
    
    # Примерный парсинг системы уравнений
    system = []
    for eq in equations:
        terms = [float(i) for i in eq.split()]
        system.append(terms)

    # Проверяем выбранный метод и вызываем соответствующую функцию
    if method == 'gauss':
        logging.debug("Gauss method selected.")
        solution_steps = gauss_method(system)
    elif method == 'cramer':
        logging.debug("Cramer method selected.")
        solution_steps = cramer_method(system)
    elif method == 'lu':
        logging.debug("LU decomposition method selected.")
        solution_steps = lu_decomposition(system)
    elif method == 'jordan':
        logging.debug("Jordan-Gauss method selected.")
        solution_steps = jordan_gauss_method(system)
    elif method == 'inverse_matrix':
        logging.debug("Inverse Matrix method selected.")
        solution_steps = inverse_matrix_method(system)
    elif method == 'jacobi':
        logging.debug("Jacobi method selected.")
        solution_steps = jacobi_method(system)
    else:
        logging.error(f"Unrecognized method: {method}")
        return "Метод не реализован"
    
    return "<br>".join(solution_steps)

# Метод Гаусса
def gauss_method(equations):
    augmented_matrix = Matrix(equations)
    steps = []
    
    steps.append(f"Начальная матрица: {augmented_matrix}")
    
    row, col = augmented_matrix.shape
    for i in range(row):
        augmented_matrix[i, :] = augmented_matrix[i, :] / augmented_matrix[i, i]
        steps.append(f"Шаг {i+1}: Нормализуем строку {i+1}: {augmented_matrix}")
        
        for j in range(i+1, row):
            factor = augmented_matrix[j, i]
            augmented_matrix[j, :] = augmented_matrix[j, :] - factor * augmented_matrix[i, :]
            steps.append(f"Шаг {i+1}.{j}: Зануляем строку {j+1}: {augmented_matrix}")
    
    return steps

# Метод Жордана-Гаусса
def jordan_gauss_method(equations):
    augmented_matrix = Matrix(equations)
    steps = [f"Начальная матрица: {augmented_matrix}"]
    
    row, col = augmented_matrix.shape
    for i in range(row):
        augmented_matrix[i, :] = augmented_matrix[i, :] / augmented_matrix[i, i]
        steps.append(f"Шаг {i+1}: Нормализуем строку {i+1}: {augmented_matrix}")
        
        for j in range(row):
            if i != j:
                factor = augmented_matrix[j, i]
                augmented_matrix[j, :] = augmented_matrix[j, :] - factor * augmented_matrix[i, :]
                steps.append(f"Шаг {i+1}.{j}: Зануляем строку {j+1}: {augmented_matrix}")
    
    return steps

# Метод обратной матрицы
def inverse_matrix_method(equations):
    augmented_matrix = Matrix(equations)
    A = augmented_matrix[:, :-1]
    b = augmented_matrix[:, -1]
    
    steps = [f"Начальная матрица A: {A}", f"Столбец b: {b}"]
    
    try:
        A_inv = A.inv()
        x = A_inv * b
        steps.append(f"Обратная матрица A⁻¹: {A_inv}")
        steps.append(f"Решение системы: x = {x}")
    except ValueError:
        steps.append("Обратная матрица не существует, система не имеет решения.")
    
    return steps

# Метод Якоби
def jacobi_method(equations, tol=1e-10, max_iterations=100):
    A = np.array([eq[:-1] for eq in equations])
    b = np.array([eq[-1] for eq in equations])
    x = np.zeros_like(b)
    D = np.diag(A)
    R = A - np.diагflat(D)
    
    steps = [f"Начальная матрица A: {A}", f"Вектор b: {b}", f"Инициализация: x = {x}"]
    
    for it in range(max_iterations):
        x_new = (b - np.dot(R, x)) / D
        steps.append(f"Итерация {it+1}: x = {x_new}")
        
        if np.linalg.norm(x_new - x) < tol:
            steps.append(f"Система решена за {it+1} итераций: x = {x_new}")
            return steps
        
        x = x_new
    
    steps.append(f"Метод Якоби не сошелся за {max_iterations} итераций.")
    return steps

# Метод Крамера
def cramer_method(equations):
    augmented_matrix = Matrix(equations)
    A = augmented_matrix[:, :-1]
    b = augmented_matrix[:, -1]
    
    det_A = A.det()
    steps = [f"Начальная матрица A: {A}", f"Определитель матрицы A: {det_A}"]
    
    if det_A == 0:
        steps.append("Система не имеет уникальных решений, так как определитель матрицы равен нулю.")
        return steps
    
    solutions = []
    
    for i in range(A.shape[1]):
        A_i = A.copy()
        A_i[:, i] = b
        det_A_i = A_i.det()
        solution = det_A_i / det_A
        solutions.append(solution)
        steps.append(f"Определитель A_{i+1}: {det_A_i}")
        steps.append(f"x_{i+1} = {solution}")
    
    steps.append(f"Решение системы: {solutions}")
    return steps

# LU-разложение
def lu_decomposition(equations):
    augmented_matrix = np.array(equations)
    A = augmented_matrix[:, :-1]
    b = augmented_matrix[:, -1]
    
    P, L, U = lu(A)
    
    steps = [f"Матрица A: {A}", f"LU-разложение: L = {L}, U = {U}"]
    
    y = np.linalg.solve(L, b)
    steps.append(f"Решение системы L * y = b: y = {y}")
    
    x = np.linalg.solve(U, y)
    steps.append(f"Решение системы U * x = y: x = {x}")
    
    return steps

if __name__ == '__main__':
    app.run(debug=True)
