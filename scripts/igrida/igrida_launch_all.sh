SCRIPT_NAME="igrida_launch.sh"

if [ $# -ne 3 ]
then
	echo "usage: out hours mitsuba"
	exit 1
fi

out=$1
hours=$2
mitsuba=$3

sh $SCRIPT_NAME "bathroom" $out $hours $mitsuba
sh $SCRIPT_NAME "bookshelf" $out $hours $mitsuba
sh $SCRIPT_NAME "bootle" $out $hours $mitsuba
sh $SCRIPT_NAME "kitchen" $out $hours $mitsuba
sh $SCRIPT_NAME "cbox" $out $hours $mitsuba
sh $SCRIPT_NAME "pmbox" $out $hours $mitsuba
sh $SCRIPT_NAME "veach-lamp" $out $hours $mitsuba
