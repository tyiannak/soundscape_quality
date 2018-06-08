# soundscape_quality
Data handling and baseline approach for soundscape quality estimation

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

## Run regression test on audio feature statistics
```
python test_features.py -i features
```
