# mamba.c

<p align="center">
  <img src="assets/mamba-c.png" width="300" height="300" alt="Мамба C">
</p>

<p align="right"><a href="https://github.com/kroggen/mamba.c/blob/learning/README.md">English</a> | <a href="https://github.com/kroggen/mamba.c/blob/learning/README-zh.md">中文</a> | <a href="https://github.com/kroggen/mamba.c/blob/learning/README-ja.md">日本語</a></p>

Вывод моделей Mamba 1, 2 & 3 на чистом C

Вдохновлен и использует код из [llama2.c](https://github.com/karpathy/llama2.c)

Это реализует только рекуррентный режим Mamba SSM

Вы можете сравнить его с [соответствующей реализацией на pytorch](https://github.com/kroggen/mamba-cpu/tree/recurrent-only)

Нет поддержки для пакетов. Код минимален для обучения.

Тем не менее, он быстрее, чем pytorch на CPU!!!


## Быстрый Старт

Как только веса модели Mamba-3 будут публично выпущены (см. [Модели](#модели) ниже):

```
python3 tokenizer.py
python3 export.py state-spaces/mamba3-130m model.bin
make fast
./mamba model.bin -n 20 -i "Customer Support should" -t 0.0
```
Python используется только для экспорта токенизатора и модели в более простой формат (требуется transformers и pytorch).

## Модели

> **Примечание:** По состоянию на март 2026 года, веса модели Mamba-3 еще не были публично выпущены.
> Статья ([arXiv:2603.15569](https://arxiv.org/abs/2603.15569)) была отправлена 16 марта 2026 года.
> Организация [state-spaces](https://huggingface.co/state-spaces) на HuggingFace в настоящее время размещает только чекпоинты Mamba-1 и Mamba-2.
> Следите за этой страницей для будущих релизов Mamba-3.

Когда веса станут доступны, скрипт экспорта ожидает стандартный чекпоинт HuggingFace с
макетом `backbone.layers.N.mixer.*` / `backbone.layers.N.mlp.*`, используемым в `mamba3.py`.
Затем вы можете запустить:

```
python3 export.py state-spaces/mamba3-130m model.bin
```

Или вручную:

```
python3 export.py /path/to/local/mamba3-model model.bin
```

## Внутреннее Состояние

Поскольку это рекуррентная модель, можно сохранить внутреннее состояние и затем вернуться к нему позже

Чтобы получить копию внутреннего состояния:

```c
  int state_size;
  char* state = get_internal_state(mamba, &state_size);
```

Чтобы установить внутреннее состояние:

```c
  set_internal_state(mamba, state, state_size);
```


## Ветки

Код доступен в 3 версиях, каждая на отдельной ветке:

* `learning` - очень базовая
* `fused` - объединение базовых функций в более крупные (вы можете сравнить их)
* `cuda` - простая реализация на GPU, легкая для понимания

Также доступен код для Mamba 2:

* `mamba2-learning` - очень базовая ([сравнить с mamba1](https://github.com/kroggen/mamba.c/compare/learning..mamba2-learning))
* `mamba2-fused` - объединённые функции ([сравнить с learning](https://github.com/kroggen/mamba.c/compare/mamba2-learning..mamba2-fused) | [сравнить с mamba1](https://github.com/kroggen/mamba.c/compare/fused..mamba2-fused))

А также для Mamba 3 (ICLR 2026):

* `mamba3-learning` - очень базовая ([сравнить с mamba2](https://github.com/kroggen/mamba.c/compare/mamba2-learning..mamba3-learning))
* `mamba3-fused` - объединённые функции ([сравнить с learning](https://github.com/kroggen/mamba.c/compare/mamba3-learning..mamba3-fused) | [сравнить с mamba2](https://github.com/kroggen/mamba.c/compare/mamba2-fused..mamba3-fused))

Ключевые изменения Mamba-3 по сравнению с Mamba-2:
- **Трапецеидальная дискретизация**: `h_t = α*h_{t-1} + β*B̄_{t-1}x_{t-1} + γ*B̄_t*x_t` (требуется отслеживание `prev_Bx`)
- **Зависимый от данных RoPE**: B и C вращаются накопленными углами, полученными из входных θ и размера шага Δ
- **QK-нормализация**: RMSNorm применяется к B и C после проекции (заменяет норму выхода с гейтом RMSNorm)
- **Обучаемый смещение BC**: смещение, специфичное для головы, добавляется к B и C после QK-norm, инициализируется единицами
- **Без короткой свёртки**: трапецеидальное правило + смещение делает conv1d ненужным
- **Архитектура в стиле Llama**: каждый слой — это `RMSNorm → SSM → остаток → RMSNorm → SwiGLU MLP → остаток`


## Примечания

Токенизатор может потребовать некоторой доработки для специальных символов

Не стесняйтесь вносить свой вклад и отправлять PR



## Лицензия

MIT