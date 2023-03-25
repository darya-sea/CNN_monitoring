# CNN_monitoring
Convolutional Neural Network practice. Required python > 3.10.

## How to start
```
$ virtualenv --python=python3.10 venv
$ soutce venv/bin/activate
$ pip install -r requirements.txt
$ python main.py help
    usage: main.py <prepare|train|predict|sync|request_spot|get_spot_logs|clean_up|train_history|send_spot_logs>

    preapre example: 
        python console.py prepare
    train example: 
        python console.py train
    predict example: 
        python console.py predict "../CNN/heřmánkovec nevonný/2022_09_21 hermankovec/00257C.tif"
    sync example: 
        python console.py sync
        python console.py sync CNN
    request_spot example: 
        python console.py request_spot
    get_spot_logs example: 
        python console.py spot_logs
    clean_up example: 
        python console.py clean_up
    train_history example: 
        python console.py train_history
    send_spot_logs example: 
        python console.py tail_logs
```

Example:

```
$ python console.py prepare
Preparing data for 'ježatka kuří noha'...
Data for 'ježatka kuří noha':
 Input images: 262
 Training: 10439
 Validation: 2740

Preparing data for 'rozrazil perský'...
Data for 'rozrazil perský':
 Input images: 274
 Training: 27220
 Validation: 6791

Preparing data for 'heřmánkovec nevonný'...
Data for 'heřmánkovec nevonný':
 Input images: 145
 Training: 17666
 Validation: 4180

Preparing data for 'violka rolní'...
Data for 'violka rolní':
 Input images: 287
 Training: 12493
 Validation: 3033

Preparing data for 'mák vlčí'...
Data for 'mák vlčí':
 Input images: 162
 Training: 12859
 Validation: 2934

$ python console.py train
Found 79368 images belonging to 5 classes.
Found 19362 images belonging to 5 classes.
Epoch 1/60
   4/4961 [..............................] - ETA: 2:07:45 - loss: 1.5439 - acc: 0.2969
....
$ python console.py train_history
```

## Test
$ pip install tox
$ tox -c tests/tox.ini


## Calculations

### AWS

|Type|vCPUs|GPU Memory|Memory (GiB)|On-Demand Price/hr|Spot-price|
| --- | --- | --- | --- | --- | --- |
|g4dn.xlarge|4|16|16|$0.526|0.1978 (62%)|
|g4dn.2xlarge|8|16|32|$0.752|0.2828 (62%)|
|g5g.xlarge|4|16|8|$0.42|0.1579 (62%)|
|g5g.2xlarge|8|16|16|$0.556|0.2091 (62%)|


**Tests**

Epochs: 50

|Type|Used GPU|Used Memory|Time per epoch(min)|Total time(h)|On-Demand Price/hr|Spot-price|
| --- | --- | --- | --- | --- | --- | --- |
|g4dn.xlarge|6|3|6|5|2.63|0.989|
|g4dn.2xlarge|6|3|6|5|3.76|1.414|
|g5g.xlarge|6|3|6|5|2.1|0.7895|
|g5g.2xlarge|6|3|6|5|2.78|1.0455|

### Collab

|GPU Memory (GiB)|Memory (GiB)|Price per 100 instances|
| --- | --- | --- |
|15.0|12.7| €11.19|

**Tests**:

Epochs: 50

|Used GPU|Used Memory|Time per epoch(min)|Total time(h)|Units per hour|Total spent units|
| --- | --- | --- | --- | --- | --- |
|6|3|6|5|1.96|9.8|