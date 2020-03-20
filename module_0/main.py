import numpy as np


def game_core_v1(number):
    '''Просто угадываем на random, никак не используя информацию о больше или меньше.
       Функция принимает загаданное число и возвращает число попыток'''
    count = 0
    while True:
        count += 1
        predict = np.random.randint(1, 101)  # предполагаемое число
        if number == predict:
            return (count)  # выход из цикла, если угадали


def game_core_v2(number):
    '''Сначала устанавливаем любое random число, а потом уменьшаем или увеличиваем его в зависимости от того, больше оно или меньше нужного.
       Функция принимает загаданное число и возвращает число попыток'''
    count = 0
    predict = np.random.randint(1, 100)
    while number != predict:
        count += 1
        if number > predict:
            predict += 1
        elif number < predict:
            predict -= 1
    return (count)  # выход из цикла, если угадали


def score_game(game_core):
    '''Запускаем игру 1000 раз, чтобы узнать, как быстро игра угадывает число'''
    count_ls = []
    np.random.seed(1)  # фиксируем RANDOM SEED, чтобы ваш эксперимент был воспроизводим!
    random_array = np.random.randint(1, 101, size=(1000))
    for number in random_array:
        count_ls.append(game_core(number))
    score = int(np.mean(count_ls))
    print(f"Ваш алгоритм угадывает число в среднем за {score} попыток")
    return (score)


def game_core_v3(number):
    """Сначала устанавливаем любое random число, а потом генерируем новое в зависимости от того, больше оно или меньше нужного.
        При этом на каждом шаге генерации числа меняем аргументы функции генератора, сокращая диапазон значений функции.
       Функция принимает загаданное число и возвращает число попыток"""
    count, min_value, max_value = 0, 1, 101
    predict = np.random.randint(min_value, max_value)
    while number != predict:
        count += 1
        if number < predict:
            # Ограничиваем верхнее значение генератора
            max_value = predict
            predict = np.random.randint(min_value, max_value)
        elif number > predict:
            # Ограничиваем нижнее значение генератора
            min_value = predict
            predict = np.random.randint(min_value, max_value)
    # Возвращаем число попыток
    return count


# запускаем
score_game(game_core_v3)
