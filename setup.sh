pip install -r requirements.txt

# Download the data
mkdir -p data/original/erg1214
svn co http://svn.delph-in.net/erg/tags/1214/ data/original/erg1214

bash convert-redwoods.sh
