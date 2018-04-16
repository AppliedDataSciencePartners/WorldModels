git clone https://github.com/AppliedDataSciencePartners/WorldModels.git

sudo apt-get install python-pip
sudo pip install virtualenv
sudo pip install virtualenvwrapper
export WORKON_HOME=~/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh
mkvirtualenv --python=/usr/bin/python3 worldmodels

sudo apt-get install cmake swig python3-dev zlib1g-dev libopenmpi-dev python-opengl mpich xvfb xserver-xephyr vnc4server

cd WorldModels
pip install -r requirements.txt 

