
Note: trotting currently uses motor position controller link damping (nb 6) -> what's the difference?
link dampening: tries to keep the trunk orientation stable
no limit on velocity

Note from the paper: alpha forward different from alpha backward, different threshold too (cf. paper).

Current trotting implementation uses hardcoded pre-defined positions in a state machine?

Note: gravity compensation for the modes? (who to talk to? -> davide)

IK for bert -> there is no, maybe QD would be interesting


## Trotting bang bang v1

Parameters to optimize?
- thresholds (4 and maybe 8), amplitude (4 and maybe 8)

Inputs:
TD = touch down
TO = take off
eps_tau_TD -> threshold bang bang rising  value=0.8
eps_tau_TO -> threshold bang bang falling value=0.8
tau_z (1D signal for each leg)
alpha0 -> equilibrium alpha value=0.0 (straight legs)
r0 -> same for radius value=0.15
r_hat -> amplitude bang bang value=0.3
delta_r_up -> value=-0.015 (two legs in contact offset so the two others are not)
alpha_gear0 -> for steering value=0
alpha offset HL -> to have different values for front and hinder legs (different amplitude) value=0
v_d? value=0 (desired speed for what?) left over of speed controller
kv? -> used for the amplitude? value=0 (gain for desired speed)
q -> joint angles
q_z -> joint polar coordinates

max_compression (threshold = -0.02)
find the point where you have max compression of the spring
(need to know max value)

q_reached
(q_reached = eps * theta_d - abs(q - theta_d)) (eps=0.1)
is motor position close to desired pos?

alpha_hat = 0.15 for all

Outputs:
8D desired motor pos?
or (r0, alpha) (position end effector) -> where is the convertion made? (easy to re-implement?)
after the state machine in simulink

tau to tau_z (PCA) (torque)
what is betaq? (always one? yes) -> ask Dominik
why does it depends on sign of q[i+1] - q[i] -> depends on how the leg is bent, show be consistent during one run


State machine:

Legs = (FR, FL, HL, HR) (front right, front left, rear/hind left, rear/hind right)

set_alpha: th_z_alpha = alpha +/- alpha_gear (+ offset for rear legs)

set_r0:
  0: r0
  1: r0 + delta_r_up
  2: r0 + max(0, kv * (r0 - q_z)) -> no idea (r0 + max(0, kv * (r0 - current_r)))

set_alphas:
  0: keep th_z_alpha
  1: th_z_alpha = alpha +/- alpha_gear (+ alpha_offset_HL)

set_detla_theta:
  0: delta_theta = 0
  1: delta_theta = weight * r_hat

Init: alpha=alpha0 for all legs

### 1. FR HL Pull Up (Front Right HL -> rear left? hinter links?)
extend 2 legs and update angles
r0=1
alpha = alpha0

if tau_z of FL and HR > eps_tau_TD switch to next state: (indices 1 and 3)

tau_z is zero based! -> stateflow reasons...

### 2. FL HR polar push (front left, rear right)
extend those legs r0=2 (and retract the other)

if max_compression of FL and HR  == 1? switch to next state (indices 1 and 3)

## 3. FL HR radial push
keep r0 as before but update delta theta (to 1)
alpha = alpha0 + alpha_hat (alpha TO)
set delta theta (r0 + r_hat)

switch to next state after a short time (t_stance/4)

## 4. delay leg extend
r0 = -1 for FL HR
alpha = alpha0 - alpha_hat (alpha TD)
change alpha to 1 for FR HL

then r0=2

if tau_z of FR and HL < eps_tau_TO or q_reached? switch to next state (indices 1 and 3)

## 5. FL HR pull up
r0=1
alpha=alpha0
reset delta theta

go to next state if tau_z of FR and HL > eps_tau_TD (indices 0 and 2)

## 6. FR HL polar push
r0=2

if max_compression of FR and HL  == 1? switch to next state (indices 0 and 2)

## 7. FR HL radial push
same as FL HR radial push but for FR HL
set delta theta (r0 + r_hat)

go to next state after t_stance / 4

## 8. delay leg extend
r0= -1 for FR HL
alpha = alpha0 - alpha_hat (alpha TD)
change alpha to 1 for FL HR
then r0=2

go back to first state if tau_z FR and HL < eps_tau_TO or q_reached for one of the two legs


## Trotting bang bang v2

inputs/outputs same as bang bang v1

## 1. Switch R to L
alpha=alpha0 - alpha_hat (alpha TD)
r=0
for FR HL

if tau_z of FL and HR > eps_tau_TD switch to next state:

## 2. speed control 1
change alpha TO = alpha0 + alpha_hat
alphaTO = alphaTO - kv * (v -v_d) (P controller on the speed)
alphaTD = - alphaTO

## 3. FL HR push
alpha =  alpha0 + alpha_hat (alpha TO)

if q_reached FL or HR go to next state:

## 4. switch L to R

r0=1 for FL HR
alpha =  alpha0 - alpha_hat (alpha TD)

if tau_z > eps_tau_TD for FR and HL go to next state:

## 5. Speed control

alphaTO = alphaTO - kv * (v -v_d) (P controller on the speed)
alphaTD = - alphaTO

go to next state

## 6. FR HL push
alpha =  alpha0 + alpha_hat (alpha TO)
for FR HL

if q_reached for FR or HL go back to start

(before: if tau_z for FR or HL < eps_tau_TO)
