from dataclasses import dataclass
from time import perf_counter

import numpy as np


@dataclass
class GameCore:
    ERROR_LABEL = "Неправильные значения инициализации генератора"
    number_min: int = 1
    number_max: int = 101
    experiment_count: int = 1000  # количество запусков игры
    __tries_count = 0  # счетчик попыток
    algorithm = {}  # по какому алгоритму запускать игру

    def __post_init__(self):
        self.__init_algorithm()  # инициализируем словарь алгоритмов

    def game_core_v1(self, number) -> int:
        """Просто угадываем на random, никак не используя информацию о больше или меньше.
           Функция принимает загаданное число и возвращает число попыток"""
        self.__tries_count = 0
        try:
            while True:
                self.__tries_count += 1
                predict = np.random.randint(self.number_min, self.number_max)  # предполагаемое число
                if number == predict:
                    return self.__tries_count  # выход из цикла, если угадали
        except ValueError:
            print(self.ERROR_LABEL)
            exit()

    def game_core_v2(self, number) -> int:
        """Сначала устанавливаем любое random число, а потом уменьшаем или увеличиваем его в зависимости от того, больше оно или меньше нужного.
           Функция принимает загаданное число и возвращает число попыток"""
        self.__tries_count = 0
        try:
            predict = np.random.randint(self.number_min, self.number_max)
            while number != predict:
                self.__tries_count += 1
                if number > predict:
                    predict += 1
                elif number < predict:
                    predict -= 1
        except ValueError:
            print(self.ERROR_LABEL)
            exit()
        return self.__tries_count  # выход из цикла, если угадали

    def score_game(self, game_core) -> int:
        """Запускаем игру n-раз, чтобы узнать, как быстро игра угадывает число"""
        t1 = perf_counter()
        count_ls = []
        try:
            np.random.seed(1)  # фиксируем RANDOM SEED, чтобы ваш эксперимент был воспроизводим!
            random_array = np.random.randint(self.number_min, self.number_max, size=self.experiment_count)
            for number in random_array:
                count_ls.append(game_core(number))
                score = int(np.mean(count_ls))
            print(f"Ваш алгоритм угадывает число в среднем за {score} попыток")
        except ValueError:
            print(self.ERROR_LABEL)
            exit()
        t2 = perf_counter()
        print(f'Время выполнения score_game: {t2-t1}')
        return score

    def score_game_vectorize(self, game_core) -> int:
        """Векторезированная версия. Запускаем игру n-раз, чтобы узнать, как быстро игра угадывает число"""
        t1 = perf_counter()
        count_ls = np.zeros(self.experiment_count, dtype=int)
        try:
            np.random.seed(1)  # фиксируем RANDOM SEED, чтобы ваш эксперимент был воспроизводим!
            random_array = np.random.randint(self.number_min, self.number_max, size=self.experiment_count, dtype=int)

            def f(x):
                return game_core(x)

            f_vector = np.vectorize(f)
            count_ls += f_vector(random_array)
            score = int(round(np.median(count_ls)))
            print(f"Ваш алгоритм угадывает число в среднем за {score} попыток")
        except ValueError:
            print(self.ERROR_LABEL)
            exit()
        t2 = perf_counter()
        print(f'Время выполнения score_game: {t2-t1}')
        return count_ls

    def game_core_v3(self, number) -> int:
        """Сначала устанавливаем число из середины диапазона, а потом генерируем новое в зависимости от того, больше оно или меньше нужного.
            При этом на каждом шаге генерации числа меняем аргументы функции генератора, сокращая диапазон значений функции.
           Функция принимает загаданное число и возвращает число попыток"""
        self.__tries_count, rand_min, rand_max = 1, self.number_min, self.number_max
        # Первая попытка - середина диапазона
        predict = int(self.number_max) // 2

        try:
            while number != predict:
                if number < predict:
                    # Ограничиваем верхнее значение генератора
                    rand_max = predict
                    predict = np.random.randint(rand_min, rand_max)
                elif number > predict:
                    # Ограничиваем нижнее значение генератора
                    rand_min = predict + 1
                    predict = np.random.randint(rand_min, rand_max)
                self.__tries_count += 1

        except ValueError:
            print(self.ERROR_LABEL)
            exit()

        # Возвращаем число попыток
        return self.__tries_count

    def game_core_v4(self, number) -> int:
        """Сначала устанавливаем число из середины диапазона, а потом угадываем методом деления отрезка пополам.
           Функция принимает загаданное число и возвращает число попыток"""
        low, high, self.__tries_count = self.number_min, self.number_max, 1

        # Первая попытка - середина диапазона
        predict = int(self.number_max) // 2

        try:
            while number != predict:
                # Задаем число из середины текущего диапазона
                if number < predict:
                    # Ограничиваем верхнее значение диапазона
                    high = predict
                elif number > predict:
                    # Ограничиваем нижнее значение диапазона
                    low = predict + 1
                predict = int(low + high) // 2
                self.__tries_count += 1

        except ValueError:
            print(self.ERROR_LABEL)
            exit()

        # Возвращаем число попыток
        return self.__tries_count

    def __init_algorithm(self):
        self.algorithm = {1: self.game_core_v1, 2: self.game_core_v2, 3: self.game_core_v3, 4: self.game_core_v4}

    def start_game(self, algorithm_version=4, score_type=2):
        try:
            if score_type != 2:
                self.score_game(self.algorithm[algorithm_version])
            else:
                self.score_game_vectorize(self.algorithm[algorithm_version])

        except (KeyError, TypeError) as e:
            print(f'Ошибка! {e}')
            exit()


# запускаем
new_game = GameCore()
new_game.start_game()  # по умолчанию 4 алгоритм и score_game_vectorized
