# mamba.c

<p align="center">
  <img src="assets/mamba-c.png" width="300" height="300" alt="Мамба C">
</p>

<p align="right"><a href="https://github.com/kroggen/mamba.c/blob/learning/README.md">English</a> | <a href="https://github.com/kroggen/mamba.c/blob/learning/README-zh.md">中文</a> | <a href="https://github.com/kroggen/mamba.c/blob/learning/README-ja.md">日本語</a></p>

Вывод моделей Mamba на чистом C

Вдохновлен и использует код из [llama2.c](https://github.com/karpathy/llama2.c)

Это реализует только рекуррентный режим Mamba SSM

Вы можете сравнить его с [соответствующей реализацией на pytorch](https://github.com/kroggen/mamba-cpu/tree/recurrent-only)

Нет поддержки для пакетов. Код минимален для обучения.

Тем не менее, он быстрее, чем pytorch на CPU!!!


## Быстрый Старт

```
python3 tokenizer.py
python3 export.py state-spaces/mamba-130m model.bin
make cuda
./mamba-cuda model.bin -n 20 -i "Customer Support should" -t 0.0
```
Python используется только для экспорта токенизатора и модели в более простой формат (требуется transformers и pytorch).

Вы можете выбрать другую модель на этапе экспорта

## Модели

Вы можете использовать эти модели, хранящиеся на [HuggingFace](https://huggingface.co/state-spaces):

* `state-spaces/mamba-130m`
* `state-spaces/mamba-370m`
* `state-spaces/mamba-790m`
* `state-spaces/mamba-1.4b`
* `state-spaces/mamba-2.8b`
* `state-spaces/mamba-2.8b-slimpj`

Вы можете указать имя модели в качестве аргумента для скрипта `export.py`

Обратите внимание, что скрипт экспорта загрузит модель (если она еще не загружена) в каталог кэша hugingface.

При желании вы также можете указать путь к файлу модели, если вы загрузили его вручную. Пример:

```
wget https://huggingface.co/state-spaces/mamba-130m/resolve/main/config.json?download=true -O config.json
wget https://huggingface.co/state-spaces/mamba-130m/resolve/main/pytorch_model.bin?download=true -O pytorch_model.bin
python3 export.py . model.bin
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


## Примечания

Токенизатор может потребовать некоторой доработки для специальных символов

Не стесняйтесь вносить свой вклад и отправлять PR



## Лицензия

MIT