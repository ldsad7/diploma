Обучение моделей производится в отдельной jupyter-тетрадке
jupyter_notebooks/training_models.ipynb
(мы обучали на google colab-е, поэтому там могут быть colab-специфичные куски кода).

Визуализация данных производилась в отдельной jupyter-тетрадке
jupyter_notebooks/count_statistics_and_draw_graphs.ipynb

По умолчанию предобученные модели берутся из папки ```trained_models```,
поэтому, чтобы запустить предобученную модель на тесте, её нужно положить
в эту папку.
Допустимые названия моделей:
1) BERT.pt
2) BERT_GRAN_model_ru.pt
3) BERT_JOINT_model_ru.pt
4) BERT_MULTIGRAN_model_relu_ru.pt
5) BERT_MULTIGRAN_model_sigmoid_ru.pt
6) BERT_GRAN_model.pt
7) BERT_JOINT_model.pt
8) BERT_MULTIGRAN_model_relu.pt
9) BERT_MULTIGRAN_model_sigmoid.pt
10) BERT_ru.pt

Обученные модели доступны на Google Drive: https://drive.google.com/drive/folders/1c1Hdt6a9vVXb_b4kzUpzpAAf25csuwLj?usp=sharing.

В папке ```data_ru``` лежат все размеченные тексты.

В папке ```bert``` должны лежать следующие файлы
(их можно скачать, к примеру, со страницы http://docs.deeppavlov.ai/en/master/features/pretrained_vectors.html,
RuBERT [pytorch]):
1) bert_config.json
2) pytorch_model.bin
3) vocab.txt 

Последовательность действий для запуска приложения из корневой папки:
0. python -m venv myvenv
1. python -m pip install -r requirements.txt 
2. python flask_application.py
3. --> 127.0.0.1:5000
