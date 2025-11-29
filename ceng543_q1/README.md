GloVe + BiLSTM:
python src/train.py --mode glove --rnn_type lstm --epochs 5 --batch_size 64 --out_dir models/glove_lstm

# GloVe + BiLSTM için (kayıtlı model yolunu kendi dosyana göre değiştir)
python src/extract_embeddings.py --mode glove --rnn_type lstm --model_path models/glove_lstm/best_glove_lstm.pt --out_dir outputs --out_name glove_lstm_emb



GloVe + BiGRU:
python src/train.py --mode glove --rnn_type gru --epochs 5 --batch_size 64 --out_dir models/glove_gru

DistilBERT + BiLSTM (freeze_bert ile hızlı)
python src/train.py --mode bert --rnn_type lstm --freeze_bert --epochs 3 --batch_size 16 --out_dir models/bert_lstm

DistilBERT + BiGRU:
python src/train.py --mode bert --rnn_type gru --freeze_bert --epochs 3 --batch_size 16 --out_dir models/bert_gru



