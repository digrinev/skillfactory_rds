from dataclasses import dataclass

import numpy as np


@dataclass
class GameCore:
    ERROR_LABEL = "Неправильные значения инициализации генератора"
    number_min: int = 1
    number_max: int = 101
    experiment_count: int = 1000  # количество запусков игры
    __tries_count = 0  # счетчик попыток

    def game_core_v1(self, number) -> int:
        """Просто угадываем на random, никак не используя информацию о больше или меньше.
           Функция принимает загаданное число и возвращает число попыток"""
        self.__tries_count = 0
        while True:
            self.__tries_count += 1
            predict = np.random.randint(self.number_min, self.number_max)  # предполагаемое число
            if number == predict:
                return self.__tries_count  # выход из цикла, если угадали

    def game_core_v2(self, number) -> int:
        """Сначала устанавливаем любое random число, а потом уменьшаем или увеличиваем его в зависимости от того, больше оно или меньше нужного.
           Функция принимает загаданное число и возвращает число попыток"""
        self.__tries_count = 0
        predict = np.random.randint(self.number_min, self.number_max)
        while number != predict:
            self.__tries_count += 1
            if number > predict:
                predict += 1
            elif number < predict:
                predict -= 1
        return self.__tries_count  # выход из цикла, если угадали

    def score_game(self, game_core) -> int:
        """Запускаем игру n-раз, чтобы узнать, как быстро игра угадывает число"""
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

        return score

    def game_core_v3(self, number) -> int:
        """Сначала устанавливаем число из середины диапазона, а потом генерируем новое в зависимости от того, больше оно или меньше нужного.
            При этом на каждом шаге генерации числа меняем аргументы функции генератора, сокращая диапазон значений функции.
           Функция принимает загаданное число и возвращает число попыток"""
        self.__tries_count, rand_min, rand_max = 0, self.number_min, self.number_max
        # Первая попытка - середина диапазона
        predict = int(self.number_min + self.number_max) / 2
        try:
            while number != predict:
                self.__tries_count += 1
                if number < predict:
                    # Ограничиваем верхнее значение генератора
                    rand_max = predict
                    predict = np.random.randint(rand_min, rand_max)
                elif number > predict:
                    # Ограничиваем нижнее значение генератора
                    rand_min = predict
                    predict = np.random.randint(rand_min, rand_max)
        except ValueError:
            print(self.ERROR_LABEL)
            exit()

        # Возвращаем число попыток
        return self.__tries_count


# запускаем
new_game = GameCore()
new_game.score_game(new_game.game_core_v3)
