# Дистилляция нейросетевых моделей векторного поиска

Данный репозиторий содержит код проведения экспериментов для ВКР по теме "Дистилляция нейросетевых моделей векторного поиска".

### Чекпоинты моделей:
| Модель | Количество параметров* | Размер эмбеддинга | Recall@100 | Ссылка |
| - | - | - | - | - |
| Multilingual-E5-base | 86M | 64 | 0.0.901 | [ссылка](https://disk.yandex.ru/d/b_pjem8uSANEcw) |
| Multilingual-E5-large | 303M | 512 | 0.926 | [ссылка](https://disk.yandex.ru/d/g2P0qeyQr2ZEgg) |
| Multilingual-E5-small-4l | 7M | 64 | 0.726 | [ссылка](https://disk.yandex.ru/d/LfP3aB7Z7awoWw) |
| Multilingual-E5-small-4l-distill | 7M | 64 | 0.873 | [ссылка](https://disk.yandex.ru/d/Sj_Yz4gn8jxkqA) |
| Multilingual-E5-small-4l-distill-as | 7M (86M) | 64 | 0.885 | [ссылка](https://disk.yandex.ru/d/VvraSIt_dkIHjw) |

\* Слой эмбеддингов не учитывался при подсчете количества параметров

### Запуск эксперимента:

Для воспроизведения экспериментов необходимо выполнить следующие шаги.
1. Скачать и установить python3 с официального сайта версии не ниже 3.9 (Тестирование проводилось на версии 3.9.13).
2. Установить требуемые библиотеки:
    ```bash
    pip install -r requirements.txt
    ```
3. Заполнить параметры обучения в config/condig_example.yaml, значение которых равно "...". При необходимости можно изменить и другие параметры;
4. Запустить обучение командой:
    ```bash
    python3 train.py configs/config.yaml 
    ```
    При запуске может быть полезно отключить логирование в WandB с помощью переменной окружение WANDB_MODE=disabled.
