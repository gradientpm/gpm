PATH_HTML="/home/beltegeuse/projects/GradientPM/data/html/"
PATH_SCENES="/home/beltegeuse/projects/GradientPM/data/scenes/"
PATH_REFERENCES="/home/beltegeuse/projects/GradientPM/data/references"
SCENES="bathroom bookshelf bootle cbox kitchen pmbox veach-lamp"
MTS="/home/beltegeuse/projects/GradientPM/gradient_pm_code/build/release/binaries/mitsuba"
LAYER_HTML="./results/data/html_gpm.xml"

if [ $# -ne 3 ]
then
	echo "Usage: out time outtime"
	exit 1
fi

out=$1
time=$2
outtime=$3

mkdir -p $PATH_HTML/$out/$outtime

for scene in $SCENES
do
  cp $PATH_REFERENCES"/ref_$scene.hdr" $PATH_SCENES/$scene"_scene/res$out/Ref.hdr"
  cp $PATH_SCENES/$scene"_scene/res$out/GPM_time.csv" $PATH_SCENES/$scene"_scene/res$out/GPM_L2_time.csv"
  cp $PATH_SCENES/$scene"_scene/res$out/GPT_time.csv" $PATH_SCENES/$scene"_scene/res$out/GPT_L2_time.csv"
  cp $PATH_SCENES/$scene"_scene/res$out/GBDPT_time.csv" $PATH_SCENES/$scene"_scene/res$out/GBDPT_L2_time.csv"
  rm -frv $PATH_HTML/$out/$outtime/$scene
  COMMAND="python3 run.py -m $MTS -i $PATH_SCENES -s $scene -o $out -f $time -r $PATH_REFERENCES/ref_$scene.hdr -l $LAYER_HTML -d 5 -P"
  eval $COMMAND
  mv $PATH_SCENES/$scene"_scene/html$out" $PATH_HTML/$out/$outtime/$scene
done
