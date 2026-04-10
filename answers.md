1 
a: yes these physical parameters are approximately correct since the aerial manipulator got larger rotors on the front to compensate for when lifting objects. However we would like to refactor the code to have a structured configuration to easily change parameters and maybe also amke it easily extensible for in the future that we have some builder pipeline which couples all the correct params with the configs and usd files.
b: I want you to definitely use the Newton API since I dont know for sure whether this function exists in Newton, however I also want stability, so I dont know if the simulator is capable of simulating thousands of RPM and staying stable, so maybe the workaround by applying the forces directly and only visually rotating the joints makes more sense?
c: yes this matters, however I dont know whether the old legdacy code does anything ain the correct way so everything needs to be researched again and done in the Newton way.

2
d: just use INDI with collective thrust and body rates
e: this is a bug
f: we can ignore this for now

3
g: the RL policy should actually get the collective thrust plus body rates, so this also has to be updated in the prd and any other relevant locations
h: you should use the best way which works together with Newton, the fingers are a rack and pinion system so they should open and close together. It would be nice to have some explicit control over how fast the arms can move and with what kind of tuning in real life it is also a PD controller.

4
i: quaternion, however we should be able to really easily adjust what our observations are to have a ncie closed iterative loop when testing
j: we should add normalization 
k: it should be body frame

5
l: all of these values ar enot optimized this was justg me playing around, the main goal is to just have some framework where I can realy easily add rewards and adjust rewards
m: it did not work, there were many glitches, that's also the reason I am trying to switch to this new simulator now.

6
n: I have no clue at what actual frequencies everything runs IRL, pls make sure that this is easily configurable
o: I could even run more, these are just random values for testing when I didnt run it in headless mode

7
p: flattened-osprey.usd is the main file, which is the whole system + fingers, we need to copy this file together with the unflattened one inside our repo.
q: the dof* ones are the correct configs, (look for the ones that use the flattened osprey one)

8
r: I agree, we first need to verify it in the sim before we build the RL pipeline