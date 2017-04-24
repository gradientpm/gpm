SCRIPT_NAME="run_igrida.sh"
if [ $# -ne 1 ]
then
	echo "Usage: out"
	exit 1
fi

out=$1

sh $SCRIPT_NAME "bathroom" $out
sh $SCRIPT_NAME "bookshelf" $out
sh $SCRIPT_NAME "bootle" $out
sh $SCRIPT_NAME "cbox" $out
sh $SCRIPT_NAME "kitchen" $out
sh $SCRIPT_NAME "pmbox" $out
sh $SCRIPT_NAME "veach-lamp" $out
