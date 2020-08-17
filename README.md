# Redwoods HPSG/MRS Pre-processing

## Dependencies

[PyDelphin](https://github.com/delph-in/pydelphin) version 1.4.0 

```
pip install pydelphin==1.4.0
```

## Download the data

This includes the annotated data (in tsdb/gold) and the ERG grammar.

```
mkdir -p data/original/erg1214
svn co http://svn.delph-in.net/erg/tags/1214/ data/original/erg1214
```

## Extract the data

**Syntactic annotations**: HPSG supertags (.tag) and derivation trees (.tree)

```
python src/extract-convert-mrs.py --redwoods -i data/original/erg1214/tsdb/gold/ -o data/extracted/ --extract_syntax 2> data/extracted/all.err
```

In the trees, PTB-style normalizations are applied to brackets. 

To extract a single profile, use its directory only (without --redwoods)

```
python src/extract-convert-mrs.py -i data/original/erg1214/tsdb/gold/wsj00a -o data/extracted/wsj00a -extract_syntax
``` 

**Semantic annotations**: Dependency MRS with token-level span normalizations (.dmrs in JSON format)

```
python src/extract-convert-mrs.py --redwoods -i data/original/erg1214/tsdb/gold/ -o data/extracted/ --convert_semantics --extract_semantics 2> data/extracted/all.err
```

