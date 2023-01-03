# CPG vs Reflex based controller

**TODO: Walk experiment need to be redone**
TODO: check leg indices for coupling matrix

TODO: adjust params to have more or less same step length / speed

With manual tuning, repeated experiment over 4s at 60Hz (~60cm travel):

|                               | CPG (trot)       | Reflex (trot)   | CPG (walk)      | bert (walk)     | bert (trot)     |
|-------------------------------|------------------|-----------------|-----------------|-----------------|-----------------|
| Mean Speed (m/s)              |  0.15 +/- 0.01   | 0.18 +/- 0.00   | 0.07 +/- 0.01   | 0.24 +/- 0.00   | 0.17 +/- 0.02   |
| Mean cost of transport (elec) |  0.16 +/- 0.03   | 0.14 +/- 0.00   | 0.22 +/- 0.02   | 0.10 +/- 0.01   | 0.42 +/- 0.26   |
| Mean CoT (mecha)              |  0.015 +/- 0.003 | 0.013 +/- 0.003 | 0.028 +/- 0.002 | 0.008 +/- 0.001 | 0.024 +/- 0.011 |


Note: using `python optim/eval_params.py --storage sqlite:///logs/studies.db -name greybert-real-tpe-energy --env ModesBertReal-v1 -trial 115`
note, when running for longer (2m), energy efficiency improves: 0.08 +/- 0.02

## Optimized for speed

|                               | CPG (trot)       | Reflex (trot) | CPG (walk)      |
|-------------------------------|------------------|---------------|-----------------|
| Mean Speed (m/s)              |  0.25 +/- 0.01   |               | 0.14 +/- 0.01   |
| Mean cost of transport (elec) |  0.13 +/- 0.01   |               | 0.16 +/- 0.03   |
| Mean CoT (mecha)              |  0.012 +/- 0.001 |               | 0.019 +/- 0.003 |
| Improvement                   |  60% faster      |               | 40-100% faster   |


Notes:

- using `python optim/eval_params.py --storage sqlite:///logs/studies.db --env CPGBertReal-v1 -name cpg-trot-optimize-speed-2 -trial 236 --n-eval-episodes 3 --max-episode-steps 300`
when running for longer (2m), energy efficiency improves: 0.09
- best trials optimize speed 2: 236 (fastest, 0.25 m/s, drift to the left), 158, 147 (straight gait, 0.24 m/s, not energy efficient), 188


- Reflex trot were already optimized parameters
- CPG (walk) parameters do not always transfer directly for a more slippery surface

- Using `TREADMILL_ENABLED=True python optim/eval_params.py --storage sqlite:///logs/studies.db --env CPGWalkBertReal-v1 -name cpg-walk-optimize-speed -trial 256 --n-eval-episodes 3 --max-episode-steps 300`
when running for longer (2m), energy efficiency improves: 0.07 +/- 0.02
Trial 289 is a good alternative for energy efficiency.


- trial 125 / 66  of optimize speed 2 for turning (192 to try)

Turning:
```yaml
trot:
  # The pulsations will be multiply by pi
  omega_swing: 5.4
  omega_stance: 1.1
  desired_step_len: 0.05
  ground_clearance: 0.033
  ground_penetration: 0.019
```

## Plots

Pattern
```
python custom_envs/cpg/plot.py -g trot -t 2
python custom_envs/cpg/plot.py -g fast_trot -t 2
```

On flat ground (no treadmill)
```
python3 bert_utils/visualization/plot_spring.py -i logs/spring_data_1000_Hz_trot_optimized_3_4.0s_1661856973.npz -s 1000

python3 bert_utils/visualization/plot_spring.py -i logs/spring_data_1000_Hz_trot_handtuned_2_4.0s_1661856919.npz -s 1000
```

On the treadmill:
```
python3 bert_utils/visualization/plot_spring.py -i logs/spring_data_60_Hz_trot_optimized_3_4.0s_1661178353.npz --with-tracker

python3 bert_utils/visualization/plot_spring.py -i logs/spring_data_60_Hz_trot_handtuned_2_4.0s_1661177018.npz --with-tracker
```

Paper plot:
```
python3 bert_utils/visualization/plot_spring.py -i logs/spring_data_1000_Hz_trot_optimized_3_4.0s_1661856973.npz -s 1300 -e 3300

python3 bert_utils/visualization/plot_spring.py -i logs/spring_data_1000_Hz_trot_handtuned_2_4.0s_1661856919.npz -s 1000 -e 3000

python3 bert_utils/visualization/plot_spring.py -i logs/spring_data_60_Hz_trot_optimized_3_4.0s_1661178353.npz --with-tracker -s 100 -e 220

python3 bert_utils/visualization/plot_spring.py -i logs/spring_data_60_Hz_trot_handtuned_2_4.0s_1661177018.npz --with-tracker -s 100 -e 220
```


## Optimized for speed (one param per leg)


## Optimized for energy efficiency

|                               | CPG (trot)       | Reflex (trot) | CPG (walk)      |
|-------------------------------|------------------|---------------|-----------------|
| Mean Speed (m/s)              |  0.09 +/- 0.00    |               |    |
| Mean cost of transport (elec) |  0.14 +/- 0.01    |               |    |
| Mean CoT (mecha)              |  0.011 +/- 0.000  |                |  |
| Improvement                   |  15% more efficient |               |   |

Notes:
- different solutions for different speed: 3, 210, 291, 295 from cpg-trot-optimize-energy
- using `TREADMILL_ENABLED=True python optim/eval_params.py --storage sqlite:///logs/studies.db --env CPGBertReal-v1 -name cpg-trot-optimize-energy -trial 282  --n-eval-episodes 3 --max-episode-steps 300`
when running for longer (2m), energy efficiency improves: 0.05 +/- 0.01 (60% more energy efficiency than best so far)
- fast speed for trial 3 (0.2 m/s)


Changing Kp, Kd PID joint
```python
def update_pid_gains(self):
    servos = [
        "servo_right_front_hip",
        "servo_right_front_knee",
        "servo_left_front_hip",
        "servo_left_front_knee",
        "servo_left_hind_hip",
        "servo_left_hind_knee",
        "servo_right_hind_hip",
        "servo_right_hind_knee",
    ]

    for servo in servos:
        svc = ln_client.get_service("%s.key_value.write" % self.rks[servo], "robotkernel/service_provider/key_value/write")
        svc.req.keys = np.array([REG["KP"]], dtype=np.uint32)
        svc.req.values = []

        func = getattr(svc.req, "new_vector/string_packet")
        tmp = func()
        tmp.data = "110" # default: Kp=110 Kd=2 Ki=400
        svc.req.values.append(tmp)

        svc.call()
```
