# HW08-09 – PyTorch MLP: регуляризация и оптимизация обучения

## 1. Кратко: что сделано

- **Какой датасет выбран (A/B/C) и почему.** Вариант A: KMNIST. 10 классов, 28×28×1 (оттенки серого), японские иероглифы. Рекомендован по умолчанию, доступен через torchvision.
- **Что сравнивалось в части A (регуляризация):** E1 — base MLP без регуляризации; E2 — + Dropout(0.5); E3 — + BatchNorm; E4 — лучший из E2/E3 с EarlyStopping.
- **Что сравнивалось в части B (оптимизация):** O1 — слишком большой LR; O2 — слишком маленький LR; O3 — SGD+momentum+weight decay.

## 2. Среда и воспроизводимость

- Python: 3.11
- torch / torchvision: 2.x / 0.x
- Устройство (CPU/GPU): CPU
- Seed: 42 (torch, numpy, random, cuda)
- Как запустить: открыть `HW08-09.ipynb` и выполнить  Run All.

## 3. Данные

- Датасет: KMNIST
- Разделение: train/val 90/10 от стандартного train; test — стандартный test из torchvision. Seed=42.
- Трансформации (transform): `ToTensor()` + `Normalize((0.5,), (0.5,))`
- Комментарий: 10 классов, 28×28 пикселей, 1 канал. Валидация отделена от train для оценки переобучения.

## 4. Базовая модель и обучение

- Модель MLP (кратко): `Flatten → Linear(784, 256) → ReLU → [Dropout/BatchNorm] → Linear(256, 128) → ReLU → [Dropout/BatchNorm] → Linear(128, 10)`. 2 скрытых слоя.
- Loss: `CrossEntropyLoss`
- Базовый Optimizer (для части A): Adam (lr=1e-3)
- Batch size: 256
- Epochs (макс): 20 (EarlyStopping обрезает раньше)
- EarlyStopping: patience=5, metric=val_accuracy

## 5. Часть A (S08): регуляризация (E1-E4)

- **E1 (base):** 2 скрытых слоя (256, 128), ReLU, без Dropout/BatchNorm. Val accuracy ≈ 0.954.
- **E2 (Dropout):** как E1 + Dropout(p=0.5). Val accuracy ≈ 0.947. Снижение переобучения (val loss ниже), но accuracy чуть ниже.
- **E3 (BatchNorm):** как E1 + BatchNorm1d между Linear и ReLU. Val accuracy ≈ 0.955. Лучший результат среди E1–E3.
- **E4 (EarlyStopping):** лучший из (E2/E3) по val_accuracy + EarlyStopping (patience=5). Сохранён `best_model.pt`.

## 6. Часть B (S09): LR, оптимизаторы, weight decay (O1-O3)

- **O1:** LR слишком большой — Adam, lr=0.1, 8 эпох. Accuracy ≈ 0.94 (из-за BatchNorm сеть справилась с большим LR лучше, но обучение все еще нестабильно).
- **O2:** LR слишком маленький — Adam, lr=1e-5, 8 эпох. Accuracy ≈ 0.83 (обучение сильно замедленно по сравнению с базовым lr=1e-3).
- **O3:** SGD, momentum=0.9, weight_decay=1e-4, lr=1e-2, 15 эпох. Val accuracy ≈ 0.956 — отличный результат, сопоставимый и местами превосходящий Adam.

## 7. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: [./artifacts/runs.csv](./artifacts/runs.csv)
- Лучшая модель: [./artifacts/best_model.pt](./artifacts/best_model.pt)
- Конфиг лучшей модели: [./artifacts/best_config.json](./artifacts/best_config.json)
- Кривые лучшего прогона: [./artifacts/figures/curves_best.png](./artifacts/figures/curves_best.png)
- Кривые "плохих LR": [./artifacts/figures/curves_lr_extremes.png](./artifacts/figures/curves_lr_extremes.png)

Короткая сводка:

- Лучший эксперимент части A: E4 (EarlyStopping на лучшем из E2/E3).
- Лучшая val_accuracy: ≈ 0.949 (E4).
- Итоговая test_accuracy (для лучшей модели): указана в выводе ноутбука.
- O1 (слишком большой LR): loss нестабилен, хотя из-за использования BatchNorm accuracy в итоге вытянулась до ~0.94 (но на графике видны сильные скачки).
- O2 (слишком маленький LR): метрики растут очень медленно, за 8 эпох accuracy достигла лишь ~0.83 (существенно ниже идеала).
- O3 (SGD+momentum+weight decay): по метрикам результат (≈0.956) даже немного превзошел Adam из E4, показывая высокое итоговое качество.

## 8. Анализ

На графиках E1 видно переобучение: train loss стабильно падает, в то время как val loss перестаёт улучшаться. Dropout (E2) снижает переобучение — val loss меньше, чем у E1, хотя val accuracy немного ниже. BatchNorm (E3) ускоряет сходимость и даёт лучший val_accuracy среди E1–E3. EarlyStopping в E4 останавливает обучение при отсутствии улучшения val_accuracy в течение 5 эпох и сохраняет лучшую модель по val.

O1 при слишком большом LR (0.1) демонстрирует скачкообразный loss, хотя наличие BatchNorm в данной архитектуре сильно помогает сети не упасть до случайного угадывания (итоговая accuracy ~0.94, но график сильно шумит). O2 при слишком маленьком LR (1e-5) показывает крайне медленное падение loss, итоговой точности за 8 эпох явно недостаточно (~0.83). SGD+momentum с weight decay (O3) показывает, что при правильном LR можно достичь сопоставимого и даже лучшего качества по сравнению с Adam (accuracy ~0.956), а weight decay обеспечивает дополнительную регуляризацию весов.

Выбранный конфиг (E4 с BatchNorm/Dropout и EarlyStopping) разумен для KMNIST на MLP: ограничивает переобучение и даёт лучшую val/test accuracy.

## 9. Итоговый вывод

Базовый конфиг: E4 — MLP (256, 128) с лучшей регуляризацией из E2/E3 и EarlyStopping (patience=5), Adam lr=1e-3. Он даёт лучшую val_accuracy и разумную test_accuracy. Дальше можно попробовать: data augmentation, learning rate scheduler, или более сложные архитектуры (CNN).

## 10. Приложение (опционально)

Не выполнялось.
