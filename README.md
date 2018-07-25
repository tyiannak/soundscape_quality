# soundscape_quality
Data handling and baseline approach for soundscape quality estimation

Please refer to the paper [here] (https://arxiv.org/submit/2342976/view)

## Download dataset:
```
wget http://users.iit.demokritos.gr/~tyianak/soundscape_quality_dataset/features.zip
unzip features.zip -d .
```

```
wget http://users.iit.demokritos.gr/~tyianak/soundscape_quality_dataset/soundscape.csv
```

```
wget http://users.iit.demokritos.gr/~tyianak/soundscape_quality_dataset/spectrograms.zip
unzip spectrograms.zip -d .
```

## Installation of dependencies
```
pip install -r requirements.txt
```

## Run regression test on audio feature statistics (examples)
```
python test_features.py -i features -g soundscape.csv -m audio -c all
python test_features.py -i features -g soundscape.csv -m audio -c all --upsample
python test_features.py -i features -g soundscape.csv -m audio -c all --regression
python test_features.py -i features -g soundscape.csv -m audio -c all --regression --upsample

python test_features.py -i features -g soundscape.csv -m audio -c 3
python test_features.py -i features -g soundscape.csv -m audio -c 3 --upsample
python test_features.py -i features -g soundscape.csv -m audio -c 3 --regression
python test_features.py -i features -g soundscape.csv -m audio -c 3 --regression --upsample

python test_features.py -i features -g soundscape.csv -m audio -c 3_only_extremes 
python test_features.py -i features -g soundscape.csv -m audio -c 3_only_extremes --upsample
python test_features.py -i features -g soundscape.csv -m audio -c 3_only_extremes --regression
python test_features.py -i features -g soundscape.csv -m audio -c 3_only_extremes --regression --upsample
```
