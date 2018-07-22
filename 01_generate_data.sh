for i in `seq 1 8`;
do
  echo worker $i
  # on cloud:
  # xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python extract.py $1 --total_episodes $2 --time_steps $3 &
  # on macbook for debugging:
  python 01_generate_data.py $1 --total_episodes $2 --time_steps $3 &
  sleep 1.0
done

