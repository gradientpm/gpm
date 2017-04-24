PATH_SCRIPTS="/udd/agruson/Developpement/frvsense-tools/igrida/"
PATH_SCENE="/temp_dd/igrida-fs1/agruson/gradient_scenes/scenes/"

# usage: scene_name out hours
if [ $# -ne 4 ]
then
	echo "Usage: scene_name out hours mitsuba"
	exit 1
fi

scene=$1
out=$2
hours=$3
mitsuba=$4

PATH_SCENE=$PATH_SCENE$scene"_scene/"
echo "Path scene: $PATH_SCENE"
PATH_OUT=$PATH_SCENE$out"/"

if [ -d $PATH_OUT ]
then
	echo "ERROR: $PATH_OUT already exists, QUIT"
	exit 2
fi

COMMAND="python igridaOarBatch.py -m $mitsuba -s $hours:0:0 -i $PATH_SCENE$scene -o $PATH_OUT -p bermuda -j 8 -A"
echo "Executed command"
echo $COMMAND

curr=`pwd`
cd $PATH_SCRIPTS
eval $COMMAND
cd $curr
