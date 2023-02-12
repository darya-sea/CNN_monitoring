# CNN_monitoring
Convolutional Neural Network practice. Required python > 3.10.

## How to start
```
$ virtualenv --python=python3.10 venv
$ soutce venv/bin/activate
$ pip install -r requirements.txt
$ python main.py help
    usage: main.py <prepare|train|predict>

    preapre example:
        python main.py prepare ndvi
        python main.py prepare data
        python main.py prepare 
    train example: 
        python main.py train
    predict example: 
        python main.py predict "CNN/heřmánkovec nevonný/2022_09_21 hermankovec/00257C.tif"
        python main.py predict "CNN/heřmánkovec nevonný/2022_09_21 hermankovec/"
```