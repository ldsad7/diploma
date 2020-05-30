from flask_app import app
from settings import load_model

if __name__ == '__main__':
    # load_model('BERT_MULTIGRAN_model_sigmoid_ru')
    # load_model('BERT_MULTIGRAN_model_relu_ru')
    # load_model('BERT_JOINT_model_ru')
    # load_model('BERT_GRAN_model_ru')
    # load_model('BERT_ru')
    app.run(debug=True, use_reloader=False)
