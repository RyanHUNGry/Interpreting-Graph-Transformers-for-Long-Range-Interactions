# dsc180-q2

we'll have modified GPS : graphGPS
connected to explainer

result notebooks
- visualization suites
- data
run script/driver
    - params

# starting notes
each epoch runs a forward pass, computes loss based on output, and updates weights using gradient-based method
thus each epoch has its own attention matrices, 1 for each GPS layer in the architecture. Obviously, if we have L heads, each layer has L attention matrices
    # figure out how the multiheadattention forward pass works in pytorch
    # does it give an option to see all H heads' attention matrices?
    # figure out what it can output
