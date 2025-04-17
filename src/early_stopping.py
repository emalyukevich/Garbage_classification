import numpy as np
import torch

class EarlyStopping:
    """
    Реализует механизм ранней остановки обучения модели, если метрика на валидации не улучшается
    в течение заданного количества эпох (patience).

    Используется для предотвращения переобучения и сохранения лучшей модели на основе заданной метрики (например, F1).
    """

    def __init__(self, patience=5, verbose=False, delta=1e-4, path='best_model.pth'):
        """
        Инициализация параметров ранней остановки.

        Args:
            patience (int): Количество эпох без улучшений, после которого обучение будет остановлено.
            verbose (bool): Выводить ли информацию при срабатывании или сохранении.
            delta (float): Минимальное изменение метрики, считающееся улучшением.
            path (str): Путь для сохранения лучшей модели.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0                     # Счетчик неудачных эпох
        self.best_score = None              # Лучшее значение метрики
        self.early_stop = False             # Флаг остановки обучения
        self.delta = delta
        self.path = path

    def __call__(self, val_f1, model):
        """
        Метод вызывается в конце каждой эпохи.

        Args:
            val_f1 (float): Значение метрики (например, F1-score) на валидации.
            model (torch.nn.Module): Обучаемая модель.
        """
        score = val_f1

        # Первая эпоха или лучшее значение метрики
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)

        # Если улучшения нет (или оно слишком маленькое)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")

            # Достигнут предел patience — остановить обучение
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            # Улучшение метрики — сохранить модель и сбросить счетчик
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """
        Сохраняет состояние модели, если наблюдается улучшение метрики.

        Args:
            model (torch.nn.Module): Обучаемая модель.
        """
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            print(f"Validation F1 improved. Saving model to {self.path}")

