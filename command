python3 -u train.py --fold 4 \
--train_path /content/drive/MyDrive/kaggle/vietai/train.csv \
--dict_path /root/PhoBERT_base_transformers/dict.txt \
--config_path /content/drive/MyDrive/kaggle/vietai/PhoBert-Sentiment-Classification/PhoBERT_base_transformers/config.json \
--bpe-codes /root/PhoBERT_base_transformers/bpe.codes \
--pretrained_path /root/PhoBERT_base_transformers/model.bin \
--ckpt_path ./runs \
--rdrsegmenter_path /root/vncorenlp/VnCoreNLP-1.1.1.jar \
--epochs 50 \
--batch_size 24


python3 -u predict.py --fold 4 \
--train_path /content/drive/MyDrive/kaggle/vietai/train.csv \
--dict_path /root/PhoBERT_base_transformers/dict.txt \
--config_path /content/drive/MyDrive/kaggle/vietai/PhoBert-Sentiment-Classification/PhoBERT_base_transformers/config.json \
--bpe-codes /root/PhoBERT_base_transformers/bpe.codes \
--pretrained_path /root/PhoBERT_base_transformers/model.bin \
--rdrsegmenter_path /root/vncorenlp/VnCoreNLP-1.1.1.jar \
--batch_size 24 \
--weight ./runs/model_3_4.bin


python3 -u predict.py \
--train_path /content/drive/MyDrive/kaggle/vietai/test.csv \
--dict_path /root/PhoBERT_base_transformers/dict.txt \
--config_path /content/drive/MyDrive/kaggle/vietai/PhoBert-Sentiment-Classification/PhoBERT_base_transformers/config.json \
--bpe-codes /root/PhoBERT_base_transformers/bpe.codes \
--pretrained_path /root/PhoBERT_base_transformers/model.bin \
--rdrsegmenter_path /root/vncorenlp/VnCoreNLP-1.1.1.jar \
--batch_size 24 \
--weight ./runs/model_1_4.bin