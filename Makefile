
.PHONY: install train clean

install:
	pip install -r requirements.txt

train:
	python train.py

clean:
	rm -f model/*.joblib
