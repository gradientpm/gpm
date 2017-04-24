PATH_SCENE="/temp_dd/igrida-fs1/agruson/gradient_scenes/scenes/"
REFERENCES="/udd/agruson/refNormal/"
MTS="/temp_dd/igrida-fs1/agruson/mtsbin/mitsuba"
TIMES="-f 1:0:0 -f 0:30:0 -f 0:10:0"
if [ $# -ne 2 ]
then
	echo "Usage: scene_name out"
	exit 1
fi

scene=$1
out=$2

COMMAND="python3 run.py -m $MTS -i $PATH_SCENE -s $scene -o $out $TIMES -r $REFERENCES/ref_$scene.hdr -d 5 -p"
eval $COMMAND
