#

scripts_dir_path=$(dirname $0);

root_dir_path="$scripts_dir_path/../..";
cd $root_dir_path;

unzip timit.zip -d .tmp > /tmp/null;

./.venv/bin/python scripts/timit/alphabetize.py .tmp data;
./.venv/bin/python scripts/timit/process.py .tmp data;
./.venv/bin/python scripts/timit/segment.py data/train data/dev;

rm -rf .tmp;
