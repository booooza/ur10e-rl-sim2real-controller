# ur10e-rl-sim2real-controller

Run inference of RL-policy on a real UR10e robot:

```bash
python ./run.py --robot-ip 192.168.0.42 --model-path models/exported_model.onnx --max-steps 600
```

Prerequisites:
* onnxruntime
* ur_rtde
* numpy, scipy

Install ur_rtde https://sdurobotics.gitlab.io/ur_rtde/installation/installation.html
```bash
git clone https://gitlab.com/sdurobotics/ur_rtde.git
cd ur_rtde
git submodule update --init --recursive
mkdir build
cd build
cmake ..
make
sudo make install
```

Install URSim

```bash
git clone https://github.com/urrsk/ursim_docker.git
docker build ursim/e-series -t ursim --build-arg VERSION=5.24
docker run --rm -it   -p 5900:5900   -p 6080:6080   -p 29999:29999   -p 30001-30004:30001-30004 -e ROBOT_MODEL=UR10  ursim
```

The model has been exported from `rl_games` as described here:
https://github.com/Denys88/rl_games/blob/master/notebooks/train_and_export_onnx_example_continuous.ipynb