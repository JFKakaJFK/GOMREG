import gomea
import numpy as np

expression_idx = 0
ages = [44.67, 48.08, 51.83]
volumes = [0.297, 0.394, 0.450]

frv = gomea.fitness.FCREGBBO(
    expression_idx,
    ages,
    volumes,
    value_to_reach=1e-6
)
lm = gomea.linkage.Full()
rvgom = gomea.RealValuedGOMEA(
    fitness=frv,
    linkage_model=lm,
    lower_init_range=0,
    upper_init_range=1,
    max_number_of_populations=1,
    base_population_size=100,
    max_number_of_evaluations=1_000_000
)
result = rvgom.run()
result.printFinalStatistics()
