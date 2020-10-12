# Redwoods HPSG/MRS Pre-processing

## Dependencies

[PyDelphin](https://github.com/delph-in/pydelphin) version 1.4.0 

```
pip install pydelphin==1.4.0
```

**ACE Parser** (only required for parsing new sentences, not for data conversion):

Download and unzip the parser and the precompiled grammar image (ERG 1214) from the [ACE](http://sweaglesw.org/linguistics/ace/) website.

Place the ace binary in the command line executable path so that PyDelphin can access it. Place the grammar image in same folder as the grammar (by default, data/original/erg1214).

## Download the data

This includes the annotated data (in tsdb/gold) and the ERG grammar.

```
mkdir -p data/original/erg1214
svn co http://svn.delph-in.net/erg/tags/1214/ data/original/erg1214
```

## Extract the data

**Syntactic annotations**: Extracts HPSG supertags (.tag) and derivation trees (.tree).

```
python src/extract-convert-mrs.py --redwoods -i data/original/erg1214/tsdb/gold/ -o data/extracted/ --extract_syntax 2> data/extracted/all.err
```

In the trees, PTB-style normalizations are applied to brackets. 

To extract a single profile, use its directory only (without --redwoods):

```
python src/extract-convert-mrs.py -i data/original/erg1214/tsdb/gold/wsj00a -o data/extracted/wsj00a --extract_syntax
``` 

**Semantic annotations**: Extracts Dependency MRS with token-level span normalizations (.dmrs in JSON format).


```
python src/extract-convert-mrs.py --redwoods -i data/original/erg1214/tsdb/gold/ -o data/extracted/ --convert_semantics --extract_semantics 2> data/extracted/all.err
```

Extract semantics for the profile from seperate MRP EDS's instead (currently requires data not publicly available):

```
python src/extract-convert-mrs.py --mrp -i data/original/erg1214/tsdb/gold/wsj21a/ --mrp_input data/other/mrp-extended/2020/cf/validation/eds/wsj.eds --convert_semantics -o data/extracted/mrp-wsj21a --extract_semantics
```

## Parse and convert

Parse an input sentence with ACE and convert (to syntactic or semantic representation):

```
python src/parse-convert-mrs.py --extract_syntax --convert_semantics --extract_semantics
```

This feature may fail for some corner cases. 

