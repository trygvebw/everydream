import ldm.lr_scheduler as lrs

#def __init__(self, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0):
sch = lrs.EveryDreamScheduler(warm_up_steps=10, f_min=5.0e-1, f_max=1.0, f_start=1.0, steps_to_min=25, verbosity_interval=5)

for i in range(50):
    print(f"step {i}: {sch(i)}")
