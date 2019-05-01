# bash 01_generate_data.sh car_racing 8 125 300 0 5

for i in `seq 1 $2`;
do
  echo worker $i
  # on cloud with no screen:
  # xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python extract.py $1 --total_episodes $2 --time_steps $3 &

  # on local machine with screen for debugging:
  python 01_generate_data.py $1 --total_episodes $3 --time_steps $4 --render $5 --action_refresh_rate $6 &
  sleep 1.0
done

